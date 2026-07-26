import importlib
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.algorithms import PGIAVI
from src_autotest.algorithms_b import PGIAVI_B


MODEL_MODULES = ('model.intention', 'model.intention_b')
MODEL_CLASSES = ('IntentionRNN', 'IntentionLSTM', 'IntentionTransformer')


def make_model(module, class_name, gate_mode=None):
    model_class = getattr(module, class_name)
    kwargs = {
        'num_states': 9,
        'num_actions': 4,
        'num_latents': 3,
        'num_layers': 1,
        'dropout': 0.0,
    }
    if class_name == 'IntentionTransformer':
        kwargs.update(d_model=8, nhead=2)
    else:
        kwargs.update(hidden_dim=8, rnn_hidden_dim=8)
    if gate_mode is not None:
        kwargs['gate_mode'] = gate_mode
    return model_class(**kwargs).eval()


def legacy_forward(model, class_name, states, actions):
    x = model.state_embed(states) + model.action_embed(actions)
    if class_name == 'IntentionTransformer':
        return model.fc_out(model.transformer(model.pos_encoding(x)))
    recurrent_output, _ = model.rnn(x)
    return model.output_proj(recurrent_output)


def toy_problem():
    num_states = 3
    num_actions = 2
    transition = np.zeros((num_states, num_actions, num_states), dtype=np.float64)
    for state in range(num_states):
        for action in range(num_actions):
            transition[state, action, (state + action + 1) % num_states] = 1.0
    train_trajs = [
        [(0, 0, 1), (1, 1, 0), (0, 1, 2)],
        [(1, 0, 2), (2, 0, 0)],
        [(2, 1, 1), (1, 1, 0), (0, 0, 1), (1, 0, 2)],
    ]
    test_trajs = [
        [(0, 1, 2), (2, 0, 0)],
        [(1, 1, 0), (0, 0, 1), (1, 0, 2)],
    ]
    return transition, train_trajs, test_trajs


class CausalGateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.states = torch.tensor([[0, 1, 2, 3, 4]])
        self.actions = torch.tensor([[0, 1, 2, 3, 0]])

    def test_legacy_default_is_unchanged(self):
        for module_name in MODEL_MODULES:
            module = importlib.import_module(module_name)
            for class_name in MODEL_CLASSES:
                default_model = make_model(module, class_name)
                explicit_model = make_model(module, class_name, 'retrospective')
                explicit_model.load_state_dict(default_model.state_dict())
                default_output = default_model(self.states, self.actions)
                explicit_output = explicit_model(self.states, self.actions)
                manual_output = legacy_forward(
                    default_model, class_name, self.states, self.actions
                )
                self.assertEqual(
                    list(default_model.state_dict()), list(explicit_model.state_dict())
                )
                torch.testing.assert_close(default_output, explicit_output)
                torch.testing.assert_close(default_output, manual_output)

    def test_causal_and_state_history_information_sets(self):
        changed_states = self.states.clone()
        changed_states[:, 3:] = torch.tensor([[7, 8]])
        changed_actions = self.actions.clone()
        changed_actions[:, 2:] = torch.tensor([[1, 0, 3]])

        for module_name in MODEL_MODULES:
            module = importlib.import_module(module_name)
            state_embeds = torch.randn(1, 5, 8)
            action_embeds = torch.randn(1, 5, 8)
            retrospective_x = module._combine_gate_inputs(
                state_embeds, action_embeds, 'retrospective'
            )
            causal_x = module._combine_gate_inputs(
                state_embeds, action_embeds, 'causal'
            )
            torch.testing.assert_close(retrospective_x, state_embeds + action_embeds)
            torch.testing.assert_close(causal_x[:, 0], state_embeds[:, 0])
            torch.testing.assert_close(
                causal_x[:, 1:], state_embeds[:, 1:] + action_embeds[:, :-1]
            )

            for class_name in MODEL_CLASSES:
                causal_model = make_model(module, class_name, 'causal')
                original = causal_model(self.states, self.actions)
                changed = causal_model(changed_states, changed_actions)
                torch.testing.assert_close(original[:, :3], changed[:, :3])

                state_model = make_model(module, class_name, 'state_only')
                changed_all_actions = (self.actions + 1) % 4
                torch.testing.assert_close(
                    state_model(self.states, self.actions),
                    state_model(self.states, changed_all_actions),
                )

    def test_padding_does_not_change_valid_outputs(self):
        states = torch.tensor([[0, 1, 2, 0, 0]])
        actions = torch.tensor([[0, 1, 2, 0, 0]])
        changed_states = torch.tensor([[0, 1, 2, 7, 8]])
        changed_actions = torch.tensor([[0, 1, 2, 3, 1]])
        mask = torch.tensor([[True, True, True, False, False]])

        for module_name in MODEL_MODULES:
            module = importlib.import_module(module_name)
            for class_name in MODEL_CLASSES:
                model = make_model(module, class_name, 'causal')
                output = model(states, actions, mask=mask, total_length=5)
                changed = model(
                    changed_states, changed_actions, mask=mask, total_length=5
                )
                torch.testing.assert_close(output[:, :3], changed[:, :3])

    def test_retrospective_candidate_replay_matches_prefix_forward(self):
        states = torch.tensor([
            [0, 1, 2, 3],
            [4, 3, 0, 0],
        ])
        actions = torch.tensor([
            [0, 1, 2, 3],
            [3, 2, 0, 0],
        ])
        mask = torch.tensor([
            [True, True, True, True],
            [True, True, False, False],
        ])

        for module_name in MODEL_MODULES:
            module = importlib.import_module(module_name)
            model = make_model(module, 'IntentionRNN', 'retrospective')
            candidate = model.candidate_action_logits(states, actions, mask=mask)
            reference = model(states, actions, mask=mask, total_length=4)
            observed = candidate.gather(
                2,
                actions[:, :, None, None].expand(-1, -1, 1, 3),
            ).squeeze(2)
            torch.testing.assert_close(
                observed[mask], reference[mask], atol=1e-6, rtol=0
            )
            changed_future_actions = actions.clone()
            changed_future_actions[:, 2:] = (changed_future_actions[:, 2:] + 1) % 4
            changed_future = model.candidate_action_logits(
                states, changed_future_actions, mask=mask
            )
            torch.testing.assert_close(
                candidate[:, :2], changed_future[:, :2], atol=1e-6, rtol=0
            )
            changed_padding_states = states.clone()
            changed_padding_actions = actions.clone()
            changed_padding_states[1, 2:] = torch.tensor([7, 8])
            changed_padding_actions[1, 2:] = torch.tensor([1, 3])
            changed_padding = model.candidate_action_logits(
                changed_padding_states, changed_padding_actions, mask=mask
            )
            torch.testing.assert_close(
                candidate[1, :2], changed_padding[1, :2], atol=1e-6, rtol=0
            )

            for row in range(states.shape[0]):
                valid_length = int(mask[row].sum())
                for step in range(valid_length):
                    for action in range(4):
                        prefix_actions = actions[row, :step + 1].clone()
                        prefix_actions[-1] = action
                        expected = model(
                            states[row, :step + 1].unsqueeze(0),
                            prefix_actions.unsqueeze(0),
                        )[0, step]
                        torch.testing.assert_close(
                            candidate[row, step, action],
                            expected,
                            atol=1e-6,
                            rtol=0,
                        )

    def test_retrospective_action_score_is_exactly_normalized(self):
        module = importlib.import_module('model.intention')
        scorer = PGIAVI.__new__(PGIAVI)
        scorer.num_latents = 2
        scorer.num_actions = 4
        scorer.gate_mode = 'retrospective'
        scorer.retrospective_score_mode = 'action_normalized'
        scorer.target_intention_net = module.IntentionRNN(
            num_states=3,
            num_actions=4,
            num_latents=2,
            hidden_dim=8,
            rnn_hidden_dim=8,
            dropout=0.0,
            gate_mode='retrospective',
        ).eval()
        states = torch.tensor([0, 1, 2, 1])
        actions = torch.tensor([0, 2, 1, 3])
        traj = [
            (int(state), int(action), int(state))
            for state, action in zip(states, actions)
        ]
        agents = [
            SimpleNamespace(q=np.array([
                [2.0, -1.0, 0.5, 0.0],
                [0.0, 1.0, -2.0, 0.5],
                [1.0, 0.0, 0.5, -1.0],
            ])),
            SimpleNamespace(q=np.array([
                [-1.0, 2.0, 0.0, 0.5],
                [1.0, -1.0, 0.5, 0.0],
                [0.0, 0.5, -1.0, 2.0],
            ])),
        ]
        legacy_log_pi = scorer.get_log_pi(traj, agents)
        reference_log_gamma, reference_log_f, legacy_log_joint = (
            scorer.intention_mapping(states, actions, legacy_log_pi)
        )
        state_before = {
            name: value.detach().clone()
            for name, value in scorer.target_intention_net.state_dict().items()
        }
        q_before = [agent.q.copy() for agent in agents]
        result = scorer.retrospective_action_scores(
            states,
            actions,
            agents,
            reference_log_f,
            reference_log_gamma,
        )
        for name, value in scorer.target_intention_net.state_dict().items():
            torch.testing.assert_close(value, state_before[name])
        for agent, expected_q in zip(agents, q_before):
            np.testing.assert_array_equal(agent.q, expected_q)
        torch.testing.assert_close(
            torch.as_tensor(result['observed_gate']),
            torch.exp(reference_log_f),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            torch.as_tensor(result['observed_responsibility']),
            torch.exp(reference_log_gamma),
            atol=1e-6,
            rtol=0,
        )
        candidate_energy = torch.as_tensor(result['candidate_log_energy'])
        log_normalizer = torch.as_tensor(result['log_normalizer'])
        normalized = candidate_energy - log_normalizer[:, None]
        torch.testing.assert_close(
            torch.logsumexp(normalized, dim=-1),
            torch.zeros(len(states), dtype=normalized.dtype),
            atol=1e-6,
            rtol=0,
        )
        observed_energy = candidate_energy.gather(
            1, actions[:, None]
        ).squeeze(1)
        legacy_score = torch.logsumexp(legacy_log_joint, dim=-1)
        epsilon_row_normalizer = np.log1p(4e-8)
        torch.testing.assert_close(
            observed_energy,
            legacy_score.to(observed_energy.dtype) - epsilon_row_normalizer,
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            torch.as_tensor(result['step_log_score']),
            observed_energy - log_normalizer,
            atol=1e-6,
            rtol=0,
        )
        self.assertLessEqual(float(np.max(result['step_log_score'])), 1e-6)
        self.assertLess(result['gate_max_error'], 1e-6)
        self.assertLess(result['responsibility_max_error'], 1e-6)
        self.assertLess(result['normalization_max_error'], 1e-6)
        invalid_agents = [
            SimpleNamespace(q=agent.q.copy()) for agent in agents
        ]
        invalid_agents[0].q[0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, 'Non-finite'):
            scorer.retrospective_action_scores(
                states,
                actions,
                invalid_agents,
                reference_log_f,
                reference_log_gamma,
            )

    def test_k1_and_action_independent_gate_have_unit_partition(self):
        module = importlib.import_module('model.intention')
        states = torch.tensor([0, 1, 2])
        actions = torch.tensor([0, 2, 1])

        for num_latents in (1, 2):
            scorer = PGIAVI.__new__(PGIAVI)
            scorer.num_latents = num_latents
            scorer.num_actions = 4
            scorer.gate_mode = 'retrospective'
            scorer.retrospective_score_mode = 'action_normalized'
            scorer.target_intention_net = module.IntentionRNN(
                num_states=3,
                num_actions=4,
                num_latents=num_latents,
                hidden_dim=8,
                rnn_hidden_dim=8,
                dropout=0.0,
                gate_mode='retrospective',
            ).eval()
            if num_latents > 1:
                scorer.target_intention_net.action_embed.weight.data.zero_()
            agents = [
                SimpleNamespace(q=np.array([
                    [2.0, -1.0, 0.5, 0.0],
                    [0.0, 1.0, -2.0, 0.5],
                    [1.0, 0.0, 0.5, -1.0],
                ]))
                for _ in range(num_latents)
            ]
            traj = [
                (int(state), int(action), int(state))
                for state, action in zip(states, actions)
            ]
            legacy_log_pi = scorer.get_log_pi(traj, agents)
            reference_log_gamma, reference_log_f, _ = scorer.intention_mapping(
                states, actions, legacy_log_pi
            )
            result = scorer.retrospective_action_scores(
                states,
                actions,
                agents,
                reference_log_f,
                reference_log_gamma,
            )
            np.testing.assert_allclose(
                result['log_normalizer'], 0.0, atol=1e-6, rtol=0
            )

    def test_uniform_policy_and_current_action_cheating_sanity(self):
        module = importlib.import_module('model.intention')
        scorer = PGIAVI.__new__(PGIAVI)
        scorer.num_latents = 4
        scorer.num_actions = 4
        scorer.gate_mode = 'retrospective'
        scorer.retrospective_score_mode = 'action_normalized'
        scorer.target_intention_net = module.IntentionRNN(
            num_states=3,
            num_actions=4,
            num_latents=4,
            hidden_dim=8,
            rnn_hidden_dim=8,
            dropout=0.0,
            gate_mode='retrospective',
        ).eval()
        states = torch.tensor([0, 1, 2])
        actions = torch.tensor([0, 2, 1])
        traj = [
            (int(state), int(action), int(state))
            for state, action in zip(states, actions)
        ]
        uniform_agents = [
            SimpleNamespace(q=np.zeros((3, 4))) for _ in range(4)
        ]
        legacy_log_pi = scorer.get_log_pi(traj, uniform_agents)
        reference_log_gamma, reference_log_f, _ = scorer.intention_mapping(
            states, actions, legacy_log_pi
        )
        result = scorer.retrospective_action_scores(
            states,
            actions,
            uniform_agents,
            reference_log_f,
            reference_log_gamma,
        )
        np.testing.assert_allclose(
            result['step_log_score'], -np.log(4), atol=1e-6, rtol=0
        )

        cheating_log_gate = torch.full((4, 4), -torch.inf)
        cheating_log_policy = torch.full((4, 4), -torch.inf)
        cheating_log_gate.fill_diagonal_(0.0)
        cheating_log_policy.fill_diagonal_(0.0)
        candidate_energy = torch.logsumexp(
            cheating_log_gate + cheating_log_policy.T, dim=-1
        )
        torch.testing.assert_close(
            candidate_energy,
            torch.zeros(4),
            atol=0,
            rtol=0,
        )
        normalized = candidate_energy - torch.logsumexp(
            candidate_energy, dim=-1
        )
        torch.testing.assert_close(
            normalized,
            torch.full((4,), -np.log(4)),
            atol=1e-6,
            rtol=0,
        )

    def test_predictive_policy_is_normalized(self):
        actions = 4
        agents = [
            SimpleNamespace(q=np.array([[20.0, -20.0, 0.0, 1.0]])),
            SimpleNamespace(q=np.array([[-10.0, 10.0, 0.0, -1.0]])),
        ]
        scorer = PGIAVI.__new__(PGIAVI)
        scorer.num_latents = 2
        scorer.gate_mode = 'causal'
        log_gate = torch.log_softmax(torch.tensor([0.3, -0.7]), dim=0)
        log_action_probs = []
        for action in range(actions):
            log_policy = scorer.get_log_pi([(0, action, 0)], agents)[:, 0]
            log_action_probs.append(torch.logsumexp(log_gate + log_policy, dim=0))
            torch.testing.assert_close(
                log_policy,
                torch.stack([
                    torch.log_softmax(torch.as_tensor(agent.q).float(), dim=-1)[0, action]
                    for agent in agents
                ]),
            )
        torch.testing.assert_close(
            torch.logsumexp(torch.stack(log_action_probs), dim=0),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )

    def test_batched_fit_contract(self):
        transition, train_trajs, test_trajs = toy_problem()
        for gate_mode in ('retrospective', 'causal', 'state_only'):
            set_seed = 17
            np.random.seed(set_seed)
            torch.manual_seed(set_seed)
            model = PGIAVI_B(
                num_latents=2,
                num_states=3,
                num_actions=2,
                train_trajs=train_trajs,
                test_trajs=test_trajs,
                P=transition,
                discount=0.9,
                hidden_dim=8,
                rnn_hidden_dim=8,
                dropout=0.0,
                reg_weight=0.0,
                kl_weight=0.0,
                num_epochs=1,
                loss_threshold=0.0,
                max_iterations=1,
                gate_mode=gate_mode,
            )
            scores, outputs, masks, _ = model.fit()
            self.assertEqual(scores['iterations'], 1)
            self.assertEqual(scores['stop_reason'], 'max_iterations')
            for split in ('train', 'test'):
                legacy_gate = np.asarray(outputs[split])
                gate = np.asarray(outputs[f'gate_{split}'])
                responsibility = np.asarray(outputs[f'responsibility_{split}'])
                mask = masks[split]
                np.testing.assert_allclose(gate[mask].sum(axis=-1), 1.0, atol=1e-6)
                np.testing.assert_allclose(
                    responsibility[mask].sum(axis=-1), 1.0, atol=1e-6
                )
                np.testing.assert_allclose(gate[~mask], 0.0)
                np.testing.assert_allclose(responsibility[~mask], 0.0)
                np.testing.assert_allclose(legacy_gate[~mask], 1.0)
                if gate_mode != 'retrospective':
                    self.assertLessEqual(
                        max(np.max(x) for x in outputs[f'step_log_score_{split}']),
                        1e-6,
                    )
            if gate_mode == 'retrospective':
                self.assertAlmostEqual(
                    scores['test_step_mean'],
                    -0.7178655862808228,
                    delta=1e-5,
                )
                np.testing.assert_allclose(
                    np.concatenate(outputs['step_log_score_test']),
                    np.array([
                        -1.0982189178466797,
                        -0.7001284956932068,
                        -0.6921038627624512,
                        -0.40584683418273926,
                        -0.6930298209190369,
                    ]),
                    atol=1e-5,
                    rtol=0,
                )
                np.testing.assert_allclose(
                    np.asarray(outputs['gate_test'])[masks['test']].reshape(-1),
                    np.array([
                        0.5497357845306396,
                        0.45026418566703796,
                        0.3175565004348755,
                        0.6824434995651245,
                        0.3340563178062439,
                        0.6659437417984009,
                        0.5558642745018005,
                        0.44413572549819946,
                        0.34801897406578064,
                        0.651980996131897,
                    ]),
                    atol=1e-4,
                    rtol=0,
                )

        np.random.seed(19)
        torch.manual_seed(19)
        normalized_model = PGIAVI_B(
            num_latents=2,
            num_states=3,
            num_actions=2,
            train_trajs=train_trajs,
            test_trajs=test_trajs,
            P=transition,
            discount=0.9,
            hidden_dim=8,
            rnn_hidden_dim=8,
            dropout=0.0,
            reg_weight=0.0,
            kl_weight=0.0,
            num_epochs=1,
            loss_threshold=0.0,
            max_iterations=1,
            gate_mode='retrospective',
            retrospective_score_mode='action_normalized',
        )
        normalized_scores, normalized_outputs, _, _ = normalized_model.fit()
        self.assertEqual(
            normalized_scores['retrospective_normalized_score_type'],
            'retrospective_action_conditional_log_probability',
        )
        self.assertLessEqual(
            max(
                np.max(scores)
                for scores in normalized_outputs[
                    'step_log_score_retrospective_normalized_test'
                ]
            ),
            1e-6,
        )
        self.assertLess(
            normalized_scores['candidate_gate_max_error'], 1e-6
        )
        self.assertLess(
            normalized_scores['candidate_responsibility_max_error'], 1e-6
        )
        self.assertLess(
            normalized_scores['candidate_normalization_max_error'], 1e-6
        )

    def test_invalid_gate_mode_is_rejected(self):
        module = importlib.import_module('model.intention')
        with self.assertRaises(ValueError):
            make_model(module, 'IntentionRNN', 'future_action')
        retrospective = make_model(
            module, 'IntentionRNN', 'retrospective'
        )
        with self.assertRaises(ValueError):
            retrospective.candidate_action_logits(
                torch.tensor([[0, 1, 2]]),
                torch.tensor([[0, 1, 2]]),
                mask=torch.tensor([[True, False, True]]),
            )
        causal = make_model(module, 'IntentionRNN', 'causal')
        with self.assertRaises(ValueError):
            causal.candidate_action_logits(
                torch.tensor([[0, 1]]), torch.tensor([[0, 1]])
            )
        transition, train_trajs, test_trajs = toy_problem()
        with self.assertRaises(ValueError):
            PGIAVI_B(
                num_latents=2,
                num_states=3,
                num_actions=2,
                train_trajs=train_trajs,
                test_trajs=test_trajs,
                P=transition,
                discount=0.9,
                gate_mode='causal',
                retrospective_score_mode='action_normalized',
            )
        with self.assertRaises(ValueError):
            PGIAVI_B(
                num_latents=2,
                num_states=3,
                num_actions=2,
                train_trajs=train_trajs,
                test_trajs=test_trajs,
                P=transition,
                discount=0.9,
                model_type='IntentionLSTM',
                retrospective_score_mode='action_normalized',
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)
