import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import os
os.environ['DISPLAY'] = ':1'
import psutil
# obtain number of CPUs on the system
# print(f'Number of CPUs: {psutil.cpu_count()}')
# before assign cpu
p = psutil.Process()
# print(f'CPU pool before assignment: {p.cpu_affinity()}')
# after assign cpu
p.cpu_affinity(range(0,32))
print(f'CPU pool after assignment: {p.cpu_affinity()}')
import argparse
import sys
import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from grid_world import grid
# from gym_miniworld import miniworld
# from world3d import world3d
import utils
import modules
from modules import (
    GridDecoder,
    GridActionEncoder,
    LinearLayer
)
from datetime import datetime
import wandb


LOGGER = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(description="vta agr parser")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--name", type=str, default="st")

    # data size
    parser.add_argument("--dataset-path", type=str, default="./data/demos")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-size", type=int, default=6)
    parser.add_argument("--init-size", type=int, default=1)
    parser.add_argument("--hil-seq-size", type=int, default=1000)

    # model size
    parser.add_argument("--state-size", type=int, default=8)
    parser.add_argument("--belief-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--latent-n", type=int, default=10)

    # observation distribution
    parser.add_argument("--obs-std", type=float, default=1.0)
    parser.add_argument("--obs-bit", type=int, default=5)

    # optimization
    parser.add_argument("--learn-rate", type=float, default=0.0005)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--max-iters", type=int, default=100000)

    # subsequence prior params
    parser.add_argument("--seg-num", type=int, default=100)
    parser.add_argument("--seg-len", type=int, default=100)

    # gumbel params
    parser.add_argument("--max-beta", type=float, default=1.0)
    parser.add_argument("--min-beta", type=float, default=0.1)
    parser.add_argument("--beta-anneal", type=float, default=100)

    # log dir
    parser.add_argument("--log-dir", type=str, default="./asset/log/")
    parser.add_argument("--save_interval", type=int, default=500)

    # coding length params
    parser.add_argument("--kl_coeff", type=float, default=1.0)
    parser.add_argument("--rec_coeff", type=float, default=1.0)
    parser.add_argument("--use_abs_pos_kl", type=float, default=0)
    parser.add_argument("--coding_len_coeff", type=float, default=1.0)
    parser.add_argument("--max_coding_len_coeff", type=float, default=0.0001)
    parser.add_argument("--use_min_length_boundary_mask", action="store_true")

    # baselines
    parser.add_argument("--ddo", action="store_true")
    return parser.parse_args()


def date_str():
    s = str(datetime.now())
    d, t = s.split(" ")
    t = "-".join(t.split(":")[:-1])
    return d + "-" + t


def set_exp_name(args):
    exp_name = args.name + "_" + date_str()
    return exp_name


def main():
    # parse arguments
    args = parse_args()

    if not args.wandb:
        os.environ["WANDB_MODE"] = "online"
    else:
        os.environ["WANDB_MODE"] = "disabled"

    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # set logger
    log_format = "[%(asctime)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stderr)

    # set size
    init_size = args.init_size

    # set device as gpu
    # device = torch.device("cuda", 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # set writer
    exp_name = set_exp_name(args)

    wandb.init(
        project="Stage_1",
        entity="zichang_team",
        name=exp_name,
        sync_tensorboard=False,
        settings=wandb.Settings(start_method="fork"),
    )

    LOGGER.info("EXP NAME: " + exp_name)
    LOGGER.info(">" * 80)
    LOGGER.info(args)
    LOGGER.info(">" * 80)

    # load dataset
    if "compile" in args.dataset_path:
        train_loader, test_loader = utils.compile_loader(args.batch_size)
        action_encoder = GridActionEncoder(
            action_size=train_loader.dataset.action_size,
            embedding_size=args.belief_size,
        )
        encoder = modules.CompILEGridEncoder(feat_size=args.belief_size)
        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
        output_normal = True
    elif "miniworld" in args.dataset_path:
        train_loader, test_loader = utils.miniworld_loader(args.batch_size)
        action_encoder = GridActionEncoder(
            action_size=train_loader.dataset.action_size,
            embedding_size=args.belief_size,
        )
        encoder = modules.MiniWorldEncoderPano(input_dim=3)
        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
        output_normal = False
    elif "cheetah" in args.dataset_path:
        train_loader, test_loader = utils.hil_loader(args.batch_size, args.hil_seq_size)
        action_encoder = LinearLayer(
            input_size=train_loader.dataset.action_size,
            output_size=args.belief_size)
        encoder = LinearLayer(
            input_size=train_loader.dataset.obs_size,
            output_size=args.belief_size)
        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
        output_normal = True
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder")
    elif "cartpole" in args.dataset_path:
        train_loader, test_loader = utils.cartpole_loader(args.batch_size, args.hil_seq_size)
        action_encoder = LinearLayer(
            input_size=train_loader.dataset.action_size,
            output_size=args.belief_size)
        encoder = LinearLayer(
            input_size=train_loader.dataset.obs_size,
            output_size=args.belief_size)
        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
        output_normal = True
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn")
    elif "lunar" in args.dataset_path:
        train_loader, test_loader = utils.lunar_loader(args.batch_size, args.hil_seq_size)
        action_encoder = LinearLayer(
            input_size=train_loader.dataset.action_size,
            output_size=args.belief_size)
        encoder = LinearLayer(
            input_size=train_loader.dataset.obs_size,
            output_size=args.belief_size)
        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
        output_normal = True
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder")
    elif "hopper" in args.dataset_path:
        train_loader, test_loader = utils.hopper_loader(args.batch_size, args.hil_seq_size)
        action_encoder = LinearLayer(
            input_size=train_loader.dataset.action_size,
            output_size=args.belief_size)
        encoder = LinearLayer(
            input_size=train_loader.dataset.obs_size,
            output_size=args.belief_size)
        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
        output_normal = True
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder")
    elif "ant" in args.dataset_path:
        train_loader, test_loader = utils.ant_loader(args.batch_size, args.hil_seq_size)
        action_encoder = LinearLayer(
            input_size=train_loader.dataset.action_size,
            output_size=args.belief_size)
        encoder = LinearLayer(
            input_size=train_loader.dataset.obs_size,
            output_size=args.belief_size)
        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
        output_normal = True
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder")
    else:
        raise ValueError(f"Unrecognize dataset_path {args.dataset_path}")

    # if args.dataset_path in ["cheetah","cartpole","lunar"]:
    if True: # TODO: improve logic
        from hssm_rl_hil import EnvModel
    else:
        from hssm_rl import EnvModel
    seq_size = train_loader.dataset.seq_size

    # init models
    use_abs_pos_kl = args.use_abs_pos_kl == 1.0
    model = EnvModel(
        action_encoder=action_encoder,
        encoder=encoder,
        decoder=decoder,
        belief_size=args.belief_size,
        state_size=args.state_size,
        num_layers=args.num_layers,
        max_seg_len=args.seg_len,
        max_seg_num=args.seg_num,
        latent_n=args.latent_n,
        kl_coeff=args.kl_coeff,
        rec_coeff=args.rec_coeff,
        use_abs_pos_kl=use_abs_pos_kl,
        coding_len_coeff=args.coding_len_coeff,
        use_min_length_boundary_mask=args.use_min_length_boundary_mask,
        ddo=args.ddo,
        output_normal=output_normal
    )
    model = nn.DataParallel(model)
    model = model.to(device)
    LOGGER.info("Model initialized")

    # init optimizer
    optimizer = Adam(params=model.parameters(), lr=args.learn_rate, amsgrad=True)

    # test data
    # if args.dataset_path not in ["cheetah","cartpole","lunar","hopper"]:
    if False: # TODO: improve logic
        pre_test_full_state_list, pre_test_full_action_list = next(iter(test_loader))
        pre_test_full_state_list = pre_test_full_state_list.to(device)
        pre_test_full_action_list = pre_test_full_action_list.to(device)
    else:
        pre_test_full_state_list, pre_test_full_action_list, pre_test_full_level_list = next(iter(test_loader))
        pre_test_full_state_list = pre_test_full_state_list.to(device)
        pre_test_full_action_list = pre_test_full_action_list.to(device)

    exp_dir = os.path.join("experiments", args.name)
    os.makedirs(exp_dir, exist_ok=True)

    # for each iter
    torch.autograd.set_detect_anomaly(False)
    b_idx = 0
    train_loss_list =[]
    while b_idx <= args.max_iters:
        # for each batch
        # if args.dataset_path not in ["cheetah", "cartpole", "lunar"]:
        if False: # TODO: improve logic
            for train_obs_list, train_action_list in train_loader:
                b_idx += 1
                # mask temp annealing
                if args.beta_anneal:
                    model.state_model.mask_beta = (
                        args.max_beta - args.min_beta
                    ) * 0.999 ** (b_idx / args.beta_anneal) + args.min_beta
                else:
                    model.state_model.mask_beta = args.max_beta

                ##############
                # train time #
                ##############
                train_obs_list = train_obs_list.to(device)
                train_action_list = train_action_list.to(device)

                # run model with train mode
                model.train()
                optimizer.zero_grad()
                results = model(
                    train_obs_list, train_action_list, seq_size, init_size, args.obs_std
                )

                if args.coding_len_coeff > 0:
                    if results["obs_cost"].mean() < 0.02:
                        model.coding_len_coeff += 0.000002
                    elif b_idx > 0:
                        model.coding_len_coeff -= 0.000002

                    model.coding_len_coeff = min(0.002, model.coding_len_coeff)
                    model.coding_len_coeff = max(0.000000, model.coding_len_coeff)
                    results["coding_len_coeff"] = model.coding_len_coeff

                # get train loss and backward update
                train_total_loss = results["train_loss"]
                train_loss_list.append(train_total_loss.detach().cpu())
                train_total_loss.backward()
                if args.grad_clip > 0.0:
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip, error_if_nonfinite=True)
                optimizer.step()

                # log
                if b_idx % 5 == 0:
                    results["grad_norm"] = grad_norm
                    train_stats, log_str, log_data = utils.log_train(results, None, b_idx)
                    # Boundaries for grid world
                    true_boundaries = train_action_list[:, init_size:-init_size] == 4
                    true_boundaries = torch.roll(true_boundaries, 1, -1)
                    true_boundaries[:, 0] = True
                    correct_boundaries = torch.logical_and(
                        results["mask_data"].squeeze(-1) == true_boundaries, true_boundaries
                    ).sum()
                    num_pred_boundaries = results["mask_data"].sum()
                    num_true_boundaries = true_boundaries.sum()
                    train_stats["train/precision"] = (
                        correct_boundaries / num_pred_boundaries
                    )
                    train_stats["train/recall"] = correct_boundaries / num_true_boundaries

                    LOGGER.info(log_str, *log_data)
                    wandb.log(train_stats, step=b_idx)

                np.set_printoptions(threshold=100000)
                torch.set_printoptions(threshold=100000)
                if b_idx % 200 == 0:
                    exp_dir = os.path.join("experiments", args.name, str(b_idx))
                    os.makedirs(exp_dir, exist_ok=True)
                    for batch_idx in range(min(train_obs_list.shape[0], 10)):
                        states = train_obs_list[batch_idx][init_size:-init_size]
                        actions = train_action_list[batch_idx][init_size:-init_size]
                        reconstructed_actions = torch.argmax(results["rec_data"], -1)[
                            batch_idx
                        ]
                        options = results["option_list"][batch_idx]
                        boundaries = results["mask_data"][batch_idx]
                        frames = []
                        curr_option = options[0]

                        for seq_idx in range(states.shape[0]):
                            # read new option if boundary is 1
                            if boundaries[seq_idx].item() == 1:
                                curr_option = options[seq_idx]

                            # panorama observation for miniworld
                            if args.dataset_path == "miniworld":
                                ################################################
                                #  Begin of miniworld specific
                                ################################################
                                f = []
                                for i in range(5):
                                    f.append(states[seq_idx][:, :, i * 3 : (i + 1) * 3])
                                frame = torch.cat(f[::-1], axis=1)
                                # frame = world3d.Render(frame.cpu().data.numpy())
                                # frame.write_text(
                                #     f"Action: {repr(miniworld.MiniWorldEnv.Actions(actions[seq_idx].item()))}")
                                # frame.write_text(
                                #     f"Reconstructed: {repr(miniworld.MiniWorldEnv.Actions(reconstructed_actions[seq_idx].item()))}")
                                if (
                                    actions[seq_idx].item()
                                    == reconstructed_actions[seq_idx].item()
                                ):
                                    frame.write_text("CORRECT")
                                else:
                                    frame.write_text("WRONG")

                                # if actions[seq_idx].item() == miniworld.MiniWorldEnv.Actions.pickup:
                                #     frame.write_text("PICKUP")
                                # else:
                                #     frame.write_text("NOT PICKUP")

                                ################################################
                                #  End of miniworld specific
                                ################################################
                            elif args.dataset_path == "compile":
                                ################################################
                                #  Begin of compile specific
                                ################################################
                                frame = grid.GridRender(10, 10)
                                # this double for loop is for visualization
                                for x in range(10):
                                    for y in range(10):
                                        obj = np.argmax(
                                            states[seq_idx][x][y].cpu().data.numpy()
                                        )
                                        if (
                                            obj == grid.ComPILEObject.num_types()
                                            or states[seq_idx][x][y][
                                                grid.ComPILEObject.num_types()
                                            ]
                                        ):
                                            frame.draw_rectangle(
                                                np.array((x, y)), 0.9, "cyan"
                                            )
                                        elif obj == grid.ComPILEObject.num_types() + 1:
                                            frame.draw_rectangle(
                                                np.array((x, y)), 0.7, "black"
                                            )
                                        elif states[seq_idx][x][y][obj] == 1:
                                            frame.draw_rectangle(
                                                np.array((x, y)),
                                                0.4,
                                                grid.ComPILEObject.COLORS[obj],
                                            )
                                frame.write_text(
                                    f"Action: {repr(grid.Action(actions[seq_idx].item()))}"
                                )
                                frame.write_text(
                                    f"Reconstructed: {repr(grid.Action(reconstructed_actions[seq_idx].item()))}"
                                )
                                if (
                                    actions[seq_idx].item()
                                    == reconstructed_actions[seq_idx].item()
                                ):
                                    frame.write_text("CORRECT")
                                else:
                                    frame.write_text("WRONG")
                                ################################################
                                #  End of compile specific
                                ################################################
                            frame.write_text(f"Option: {curr_option}")
                            frame.write_text(f"Boundary: {boundaries[seq_idx].item()}")
                            frame.write_text(f"Obs NLL: {results['obs_cost'].mean()}")
                            frame.write_text(
                                f"Coding length: {results['encoding_length'].item()}"
                            )
                            frame.write_text(
                                f"Num reads: {results['mask_data'].sum(1).mean().item()}"
                            )
                            frames.append(frame.image())

                        save_path = os.path.join(exp_dir, f"{batch_idx}.gif")
                        frames[0].save(
                            save_path,
                            save_all=True,
                            append_images=frames[1:],
                            duration=750,
                            loop=0,
                            optimize=True,
                            quality=20,
                        )

                if b_idx % 100 == 0:
                    LOGGER.info("#" * 80)
                    LOGGER.info(">>> option list")
                    LOGGER.info("\n" + repr(results["option_list"][:10]))
                    LOGGER.info(">>> boundary mask list")
                    LOGGER.info("\n" + repr(results["mask_data"][:10].squeeze(-1)))
                    LOGGER.info(">>> train_action_list")
                    LOGGER.info("\n" + repr(train_action_list[:10]))
                    LOGGER.info(">>> argmax reconstruction")
                    LOGGER.info("\n" + repr(torch.argmax(results["rec_data"], -1)[:10]))
                    LOGGER.info(">>> diff")
                    LOGGER.info(
                        "\n"
                        + repr(
                            train_action_list[:10, 1:-1]
                            - torch.argmax(results["rec_data"][:10], -1)
                        )
                    )
                    LOGGER.info(">>> marginal")
                    LOGGER.info("\n" + repr(results["marginal"]))
                    LOGGER.info("#" * 80)
                    
                if b_idx % 10 == 0:
                    exp_dir = os.path.join("experiments", args.name)
                    torch.save(
                        model.state_model, os.path.join(exp_dir, f"model-{b_idx}.ckpt")
                    )

                #############
                # test time #
                #############
                if b_idx % 100 == 0:
                    with torch.no_grad():
                        ##################
                        # test data elbo #
                        ##################
                        model.eval()
                        results = model(
                            pre_test_full_state_list,
                            pre_test_full_action_list,
                            seq_size,
                            init_size,
                            args.obs_std,
                        )

                        # log
                        test_stats, log_str, log_data = utils.log_test(results, None, b_idx)
                        # Boundaries for grid world
                        true_boundaries = (
                            pre_test_full_action_list[:, init_size:-init_size] == 4
                        )
                        true_boundaries = torch.roll(true_boundaries, 1, -1)
                        true_boundaries[:, 0] = True
                        correct_boundaries = torch.logical_and(
                            results["mask_data"].squeeze(-1) == true_boundaries,
                            true_boundaries,
                        ).sum()
                        num_pred_boundaries = results["mask_data"].sum()
                        num_true_boundaries = true_boundaries.sum()
                        test_stats["valid/precision"] = (
                            correct_boundaries / num_pred_boundaries
                        )
                        test_stats["valid/recall"] = (
                            correct_boundaries / num_true_boundaries
                        )
                        LOGGER.info(log_str, *log_data)
                        wandb.log(test_stats, step=b_idx)
#########################################################################################################################
        else:
            for train_obs_list, train_action_list, train_level_list in train_loader:
                b_idx += 1
                # mask temp annealing
                if args.beta_anneal:
                    model.module.state_model.mask_beta = (
                        args.max_beta - args.min_beta
                    ) * 0.999 ** (b_idx / args.beta_anneal) + args.min_beta
                else:
                    model.module.state_model.mask_beta = args.max_beta

                ##############
                # train time #
                ##############
                train_obs_list = train_obs_list.to(device)
                train_action_list = train_action_list.to(device)

                # run model with train mode
                model.module.train()
                optimizer.zero_grad()
                results = model.module(
                    train_obs_list, train_action_list, seq_size, init_size, args.obs_std
                )

                if args.coding_len_coeff > 0:
                    if results["obs_cost"].mean() < 0.02:
                        model.module.coding_len_coeff += 0.00000001
                    elif b_idx > 0:
                        model.module.coding_len_coeff -= 0.00000001

                    model.module.coding_len_coeff = min(args.max_coding_len_coeff, model.module.coding_len_coeff)
                    model.module.coding_len_coeff = max(0.000000, model.module.coding_len_coeff)
                    results["coding_len_coeff"] = model.module.coding_len_coeff
                
                # get train loss and backward update
                train_total_loss = results["train_loss"]
                train_loss_list.append(train_total_loss.detach().cpu())
                train_total_loss.backward()
                if args.grad_clip > 0.0:
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.module.parameters(), args.grad_clip, error_if_nonfinite=True)
                optimizer.step()

                # get num of skills in traj
                num_skills = [len(i) for i in results["unique_z_list"]]

                # log
                if b_idx % 5 == 0:
                    results["grad_norm"] = grad_norm
                    train_stats, log_str, log_data = utils.log_train(results, None, b_idx)
                    train_stats["train/loss"] = train_total_loss.detach().cpu()
                    train_stats["num_skills"] = sum(num_skills)/len(num_skills)
                    LOGGER.info(log_str, *log_data)
                    # LOGGER.info("ep: {:08}, training loss: {}".format(b_idx,train_total_loss.detach().cpu()))
                    wandb.log(train_stats, step=b_idx)
                np.set_printoptions(threshold=100000)
                torch.set_printoptions(threshold=100000)
                    
                if b_idx % args.save_interval == 0:
                    exp_dir = os.path.join("experiments", args.name)
                    torch.save(
                        model.module.state_model, os.path.join(exp_dir, f"model-{b_idx}.ckpt")
                    )
                    
                #############
                # test time #
                #############
                if b_idx % 100 == 0:
                    with torch.no_grad():
                        ##################
                        # test data elbo #
                        ##################
                        model.module.eval()
                        results = model.module(
                            pre_test_full_state_list,
                            pre_test_full_action_list,
                            seq_size,
                            init_size,
                            args.obs_std,
                        )
                        # log
                        test_stats, log_str, log_data = utils.log_test(results, None, b_idx)
                        LOGGER.info(log_str, *log_data)
                        wandb.log(test_stats, step=b_idx)
    # offline training loss save
    # from matplotlib import pyplot as plt
    # plt.plot(train_loss_list, label='train_loss')
    # plt.legend()
    # plt.show()
    # plt.savefig("figs/" + exp_name + '.png')

    # offline save training loss curve as one line string in txt file
    # with open("txt_loss/" + exp_name + '.txt', 'w') as f:
    #     for item in train_loss_list:
    #         f.write("%s\n" % item)
    

if __name__ == "__main__":
    main()
