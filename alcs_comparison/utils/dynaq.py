# if __name__ == '__main__':
#     grid = gym.make('grid-20-v0')

#     def knowledge_calculator(env, model):
#         return 0
#         # transitions = env.env.get_transitions()
#         # nr_correct = 0
#         #
#         # for (s0, a, s1) in transitions:
#         #     if s0 in model and a in model[s0] and model[s0][a][0] == s1:
#         #         nr_correct += 1
#         #
#         # return nr_correct / len(transitions)
# #
#     q, model, metrics = dynaq(grid, episodes=100,
#                               num_states=grid.env.observation_space.n,
#                               num_actions=grid.env.action_space.n,
#                               epsilon=0.5,
#                               learning_rate=0.1,
#                               gamma=0.9,
#                               planning_steps=5,
#                               knowledge_fcn=knowledge_calculator)
# #
#     print("Finished")
#     print(metrics[3])
