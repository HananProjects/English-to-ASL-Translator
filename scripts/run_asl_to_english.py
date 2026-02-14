from core.engine import asl_to_english


def main():
    result = asl_to_english(tokens=["YOU", "GO", "WHERE"])
    print(result)


if __name__ == "__main__":
    main()
