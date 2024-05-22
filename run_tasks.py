import argparse
import tasks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    args, unknown = parser.parse_known_args()

    runner = getattr(tasks, args.task)()
    runner.run(parser)