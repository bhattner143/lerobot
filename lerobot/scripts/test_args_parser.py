import argparse

def main():
    arg_parser = argparse.ArgumentParser(
        description="Train a policy using the specified configuration."
    )

    arg_parser.add_argument(
        "--dataset_repo_id",
        type=str,
        default="lerobot/so_100_tele_op_cloth_flatening",
        help="Dataset repository ID (e.g., HF_USER/aloha_test)."
    )

    arg_parser.add_argument(
        "--policy_type",
        type=str,
        default="act",
        help="Type of policy to use (e.g., act)."
    )

    arg_parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/train/act_so_100_test",
        help="Directory to save training outputs."
    )

    arg_parser.add_argument(
        "--job_name",
        type=str,
        default="act_so_100_test",
        help="Name of the training job."
    )

    arg_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (e.g., cuda or cpu)."
    )

    arg_parser.add_argument(
        "--wandb_enable",
        default=False,
        action="store_true",
        help="Enable Weights & Biases logging."
    )

    args = arg_parser.parse_args()

    # Example: print the parsed arguments
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

if __name__ == "__main__":
    main()