"""Build the duration-error notebook through the shared experiment builder."""

from tools.build_modeling_notebooks import build_notebook


if __name__ == "__main__":
    print(f"Built {build_notebook('duration_error_mins')}")
