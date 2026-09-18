"""``python -m studio`` : lance le serveur du studio."""
import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="YuE2 Studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--reload", action="store_true", help="rechargement auto du code (développement)")
    args = parser.parse_args()
    from .paths import load_env_file
    loaded = load_env_file()                     # .env à la racine du dépôt (clé LLM…), ignoré par git
    if loaded:
        print(f"[studio] variables chargées depuis .env : {', '.join(loaded)}")
    uvicorn.run("studio.app:app", host=args.host, port=args.port, reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
