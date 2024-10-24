# AI-enabled-Index-Investing
* To chunk files in `inputs` and generate embeddings. run `python cli.py --chunk --embed --load`. By default they use semantic split.
* To generate an answer table, run `python cli.py --chat`. Change the `zipname` variable under the `chat` function in `cli.py` to specify the zipfile to ask questions to.
* To use the auto evaluator to test against "MSCI Select ESG Screened Indexes Methodology" with expected answers, run `python auto_evaluator.py --with_expected_answer`.