import traceback
from sys import stdout

def wrapLoaderWithExceptionHandler(LoaderType):
    class StreamLoaderExceptionCatcher(LoaderType):
        def run(self):
            print(">>>>> __init__() method of StreamLoaderExceptionCatcher")
            stdout.flush()
            try:
                super().run()
            except Exception:
                traceback.print_exc()
                self.data.put(None)

    return StreamLoaderExceptionCatcher
