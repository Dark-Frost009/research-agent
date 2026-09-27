"""Run public research in a disposable process with an enforced time limit."""
import json
import os
from pathlib import Path
import subprocess
import sys

from research_agent.incomplete import IncompleteResearch, ResearchInterrupted
from research_agent.models.schemas import Evidence, ResearchReport, Source
from research_agent.run_summary import RunSummary
from research_agent.ui_service import CompletedResearch
from research_agent.public_limits import RUN_SECONDS


class ResearchTimeoutError(RuntimeError):
    pass


def _communicate(process, payload, timeout):
    try:
        output, _ = process.communicate(payload, timeout=timeout)
        return output
    except subprocess.TimeoutExpired:
        raise ResearchTimeoutError(
            'Research reached the three-minute time limit and was stopped. '
            'No new report was released. Provider requests already sent may still be charged.'
        ) from None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        for pipe in (process.stdin, process.stdout):
            if pipe is not None:
                pipe.close()


def execute_question(question, settings):
    # Secrets travel through an anonymous pipe, never argv, files or logs.
    payload = json.dumps(dict(question=question,
        gemini_key=settings.llm_api_key.get_secret_value(),
        tavily_key=settings.tavily_api_key.get_secret_value(),
        model=settings.llm_model), ensure_ascii=False)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
    process = subprocess.Popen(
        [sys.executable, '-m', 'research_agent.public_worker'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding='utf-8', env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
    )
    output = _communicate(process, payload, RUN_SECONDS)
    if process.returncode == 124:
        raise ResearchTimeoutError('Research reached the three-minute time limit and was stopped. '
                                   'Provider requests already sent may still be charged.')
    if process.returncode != 0:
        raise RuntimeError('Research could not finish.')
    data = json.loads(output)
    if data['kind'] == 'partial':
        raise ResearchInterrupted(IncompleteResearch.model_validate(data['value']))
    if data['kind'] != 'completed':
        raise RuntimeError('Research could not finish.')
    value = data['value']
    result = CompletedResearch(
        report=ResearchReport.model_validate(value['report']),
        evidence=[Evidence.model_validate(e) for e in value['evidence']],
        sources=[Source.model_validate(s) for s in value['sources']],
        iterations=value['iterations'], searches=value['searches'],
        warning_count=data['warning_count'], issues=tuple(value['issues']),
        summary=RunSummary.model_validate(value['summary']) if value['summary'] else None,
    )
    for citation in result.report.citations:
        result.citation_sources(citation)
    return result
