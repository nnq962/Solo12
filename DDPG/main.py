import os
from DDPG import DDPGagent
from utils import *
from gym.envs.registration import register
import time
from datetime import datetime
import argparse


# env = NormalizedEnv(gym.make("Pendulum-v0"))


def create_unique_dir(base_name):
    counter = 0
    dir_name = base_name  # Khởi đầu với tên ban đầu

    # Kiểm tra sự tồn tại của thư mục
    while os.path.exists(dir_name):
        counter += 1
        dir_name = f"{base_name}_{counter}"  # Cập nhật tên với số đếm và dấu chấm

    # Thư mục với tên duy nhất không tồn tại, tạo nó
    os.mkdir(dir_name)
    print(f"Folder '{dir_name}' created.")

    return dir_name


def to_text(path, info):
    res_str = ''
    res_str = res_str + 'Env name: ' + 'Laikago' + '\n'
    res_str = res_str + 'Episode length: ' + str(info.episode_length) + '\n'
    res_str = res_str + 'Domain randomization: ' + str(info.domain_rand) + '\n'
    res_str = res_str + 'Curriculmn introduced at iteration: ' + str(info.curi_learn) + '\n'
    res_str = res_str + info.msg + '\n'
    fileobj = open(path, 'w')
    fileobj.write(res_str)
    fileobj.close()


def train(environment, info):

    # Create folder
    logdir = "experiments/" + str(time.strftime("%d_%m"))
    working_dir = os.getcwd()
    unique_dir_name = create_unique_dir(logdir)
    os.chdir(unique_dir_name)
    if os.path.isdir('iterations') is False:
        os.mkdir('iterations')
    log_dir = os.getcwd()
    to_text('hyperparameters', info)
    os.chdir(working_dir)

    agent = DDPGagent(environment)
    noise = OUNoise(environment.action_space)

    for episode in range(info.max_episodes):
        if info.domain_rand:
            environment.set_randomization(default=False)
        else:
            environment.randomize_only_inclines()
            # Học tập theo chương trình
        if episode > info.curilearn:
            avail_deg = [7, 9, 11, 13]
            environment.incline_deg = avail_deg[random.randint(0, 3)]
        else:
            avail_deg = [7, 9]
            environment.incline_deg = avail_deg[random.randint(0, 1)]

        state = environment.reset()
        noise.reset()
        episode_reward = 0

        for step in range(info.max_steps):
            action = agent.get_action(state)
            action = noise.get_action(action, step)
            new_state, reward, done, _ = environment.step(action)
            agent.memory.push(state, action, reward, new_state, done)

            if len(agent.memory) > info.batch_size:
                agent.update(info.batch_size)

            state = new_state
            episode_reward += reward

        #     if done:
        #         print(f"Episode: {episode},"
        #               f" Reward: {np.round(episode_reward, 2)},"
        #               f" Average Reward: {np.mean(rewards[-10:])}")
        #         break
        #
        # rewards.append(episode_reward)
        # avg_rewards.append(np.mean(rewards[-10:]))

        # Save models every 10 episodes
        if episode % 10 == 0:
            save_models_for_training(agent, f" iterations/ddpg_agent_{episode}.pth")

    # Save final models for testing
    save_models_for_testing(agent, "iterations/ddpg_agent_final.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--steps', help='Length of each episode', type=int, default=500)
    parser.add_argument('--episode_length', help='Number of episode', type=float, default=200)
    parser.add_argument('--eval_step', help='Policy evaluation after how many steps should take place', type=int,
                        default=3)
    parser.add_argument('--msg', help='Msg to save in a text file', type=str, default='')
    parser.add_argument('--batch_size', help='Batch size', type=int, default='128')
    parser.add_argument('--domain_rand', help='add domain randomization', type=bool, default=False)
    parser.add_argument('--curi_learn',
                        help='after how many iteration steps second stage of curriculum learning should start',
                        type=int, default=60)
    args = parser.parse_args()

    # Custom environments that you want to use ------------------------------------------------------------------------
    # register(id=args.env,
    #          entry_point='gym_sloped_terrain.envs.Laikago_pybullet_env:LaikagoEnv',
    #          kwargs={'gait': args.gait, 'render': False, 'action_dim': args.action_dim})
    # Todo: check it
    # -----------------------------------------------------------------------------------------------------------------

    print("==================   Start Training  ==================")
    start_time = time.time()

    # Train
    # train(environment=None, info=args)

    end_time = time.time()
    start_time_str = datetime.fromtimestamp(start_time).strftime('%H:%M %d/%m/%Y')
    end_time_str = datetime.fromtimestamp(end_time).strftime('%H:%M %d/%m/%Y')
    run_time = end_time - start_time

    print("Start time:  ", start_time_str)
    print("End time:    ", end_time_str)
    print("Run time:    ", np.round(run_time / 3600, 3), "hours")
    print("==================   End Training    ==================")
