
import sys


def get_user_input() -> str:
    """
    Read multiline user input from the terminal.
    Finish input by entering a line containing only END.
    """
    print("Enter your prompt (type END on a new line to finish):")

    lines = []

    while True:
        line = sys.stdin.readline()

        if not line:
            break

        if line.rstrip("\r\n") == "END":
            break

        lines.append(line)

    return "".join(lines).strip()
