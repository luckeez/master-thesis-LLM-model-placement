from main import main
import os

tests = [
    "test15-split.ini",
    "test15-unique.ini",
    "test18-a30.ini",
    "test-cloud.ini"
]

if __name__ == '__main__':
    for test in tests:
        print(f"Running test: {test}")
        main(complete_cluster_file_name=f"./config/{test}")
        os.rename(f"./layouts/ilp", f"./layouts/ilp_{test[:-4]}")