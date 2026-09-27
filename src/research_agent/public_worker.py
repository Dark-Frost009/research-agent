"""Private subprocess entry point; stdout is a single JSON result, never logs."""
import os
import sys
import threading


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
            request = json.loads(sys.stdin.read(8192))
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
