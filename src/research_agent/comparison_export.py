"""Portable Markdown comparison of two already-loaded history snapshots."""
from datetime import timezone
import re

from research_agent.incomplete import IncompleteResearch
from research_agent.ui_service import safe_source_url


def source_urls(result):
    return {url for source in result.sources if (url := safe_source_url(source))}


def _literal(text):
    # Keep research text literal, including HTML, images, and embedded fences.
    fence = '`' * max(3, 1 + max((len(run) for run in re.findall(r'`+', text)), default=0))
    return f'{fence}text\n{text}\n{fence}'


def comparison_markdown(first_id, first, second_id, second):
    left_urls, right_urls = source_urls(first), source_urls(second)
    questions = [run.question if isinstance(run, IncompleteResearch) else run.report.question
                 for run in (first, second)]
    lines = ['# Research run comparison',
             'Comparison of saved snapshots. No new research or verification was performed. '
             'This export does not score answer quality.',
             'Budget usage counts saved reservations, not provider billing or token totals. '
             'Source overlap uses exact available HTTP(S) URLs, not page-content equality.']
    if questions[0].strip() != questions[1].strip():
        lines.append('These runs used different questions; consider that when comparing answers.')
    lines += ['## Source differences',
              f'{len(left_urls & right_urls)} shared; {len(left_urls - right_urls)} only in first; '
              f'{len(right_urls - left_urls)} only in second.']
    for label, urls in [('Shared source URLs', left_urls & right_urls),
                        ('Only in first run', left_urls - right_urls),
                        ('Only in second run', right_urls - left_urls)]:
        lines += ['### ' + label, _literal('\n'.join(sorted(urls))) if urls else 'None.']
    for label, record_id, result, question in zip(('First run', 'Second run'),
                                                  (first_id, second_id), (first, second), questions):
        incomplete = isinstance(result, IncompleteResearch)
        created_at = result.created_at if incomplete else result.report.created_at
        depth = getattr(result, 'depth', None)
        summary = getattr(result, 'summary', None)
        lines += ['## ' + label, 'Saved entry ID:', _literal(record_id),
                  'Created: ' + created_at.astimezone(timezone.utc).isoformat(),
                  'Status: ' + ('Incomplete' if incomplete else 'Completed'),
                  '### Question', _literal(question), '### Research depth',
                  depth.summary if depth else 'Not recorded.', '### Outcome']
        if incomplete:
            lines += ['Interrupted run. No verified answer was released.', result.stop_message]
        else:
            lines.append(summary.outcome_message if summary else 'Outcome not recorded for this older report.')
        lines += ['### Answer', 'No completed answer is available for this run.' if incomplete
                  else _literal(result.report.content), '### Budget usage']
        if summary:
            lines += [summary.stop_message, '| Resource | Budget used | Limit |\n| --- | ---: | ---: |\n' +
                      '\n'.join(f"| {row['Resource']} | {row['Budget used']} | {row['Limit']} |" for row in summary.rows())]
        else:
            lines += ['Detailed budget usage and limits were not recorded.',
                      f'Recorded rounds: {result.iterations}; searches: {result.searches}.']
        lines += [f'Sources: {len(result.sources)}; evidence excerpts: {len(result.evidence)}; warnings: {result.warning_count}.',
                  '### Warnings', *[_literal(issue) for issue in result.issues]]
        if not result.issues:
            lines.append('No warning details recorded.')
        lines.append('### Sources')
        if not result.sources:
            lines.append('No sources were collected.')
        for index, source in enumerate(result.sources, 1):
            url = safe_source_url(source)
            lines += [f'#### Source {index}', _literal(source.title or source.domain),
                      _literal(url) if url else 'Source URL unavailable for comparison.',
                      'Fetch status: ' + source.fetch_status]
    return '\n\n'.join(lines) + '\n'
