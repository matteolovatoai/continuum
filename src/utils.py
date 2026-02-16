from pathlib import Path

DIR_PROJECT = Path(__file__).parent.parent.resolve()
DIR_DATA = DIR_PROJECT / "data"
DIR_AI = DIR_PROJECT / "ai-service"
DIR_WEB = DIR_PROJECT / "web"


if __name__ == "__main__":
    print(f"La home del progetto è: {DIR_PROJECT}")
    print(f"I dati sono su: {DIR_DATA}")