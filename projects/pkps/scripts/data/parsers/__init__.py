from .math import MATHParser
from .wmt_14 import WMT14Parser
from .med_qa import MEDQAParser
from .legalbench import LegalBenchParser

parsers = {
    'math'   : MATHParser,
    'wmt_14' : WMT14Parser,
    'med_qa' : MEDQAParser,
    'legalbench' : LegalBenchParser,
}