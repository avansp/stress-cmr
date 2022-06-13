import typer
import sys
from loguru import logger
from app_fcn import app as app_fcn


app = typer.Typer(add_completion=False)
app.add_typer(app_fcn, name="FCN", help="Using fully convolution network")


@app.callback()
def main(verbose: bool = typer.Option(False, help="Print debugging information")):
    """
    CLI to run STRESS CMR perfusion experiment.
    """
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    logger.add(lambda _: sys.exit(1), level="CRITICAL")


if __name__ == "__main__":
    app()
