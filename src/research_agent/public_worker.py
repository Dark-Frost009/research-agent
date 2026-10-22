"""Private subprocess entry point; stdout is a single JSON result, never logs."""
import os
import sys
import threading

# Largest legitimate request. The parent allows a 2,000-character question, two
# 512-character keys and a 150-character model. In the worst case every one of
# those characters is a control character, which JSON escapes to six characters
# (about 19,000 in all). The cap is deliberately generous but bounded: oversized
# input is rejected, never truncated into a request that parses differently.
MAX_REQUEST_CHARS = 65_536


def main():
    # Start before importing providers. Also stops an orphan after a host crash.
    from research_agent.public_limits import RUN_SECONDS
    watchdog = threading.Timer(RUN_SECONDS, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8')
    with open(os.devnull, 'w') as silent:
        os.dup2(silent.fileno(), sys.stdout.fileno())
        os.dup2(silent.fileno(), sys.stderr.fileno())
        import json
        from research_agent.public_service import visitor_settings
        from research_agent.ui_service import run_question
        from research_agent.incomplete import ResearchInterrupted
        try:
            raw = sys.stdin.read(MAX_REQUEST_CHARS + 1)
            if len(raw) > MAX_REQUEST_CHARS:
                raise ValueError('Research request is too large.')
            request = json.loads(raw)
            settings = visitor_settings(request['gemini_key'], request['tavily_key'], request['model'])
            result = run_question(request['question'], lambda _: None, settings=settings)
            response = dict(kind='completed', value=json.loads(result.json_export()),
                            warning_count=result.warning_count)
        except ResearchInterrupted as exc:
            response = dict(kind='partial', value=exc.partial.model_dump(mode='json'))
        except Exception:
            response = dict(kind='error')
        json.dump(response, protocol)
        protocol.flush()
        protocol.close()


if __name__ == '__main__':
    main()
