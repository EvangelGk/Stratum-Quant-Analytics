from logger.Messages.DirectionsMess import LIVE_STEP_0_WELCOME
from logger.Messages.MainMess import APPLICATION_TITLE
from logger.Messages.MedallionMess import BRONZE_START
from logger.Messages.FetchersMess import FETCHER_START


def test_message_constants_are_strings():
    assert isinstance(LIVE_STEP_0_WELCOME, str)
    assert isinstance(APPLICATION_TITLE, str)
    assert isinstance(BRONZE_START, str)
    assert isinstance(FETCHER_START, str)
