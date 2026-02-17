from datasets import load_from_disk


def main():
    ds = load_from_disk("data/prepped_dataset/run_1_ppls.hf")

    print(ds)
    print(ds[1])
    print(ds['ppl'][0])

if __name__ == "__main__":
    main()