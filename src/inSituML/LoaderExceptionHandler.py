import traceback


def wrapLoaderWithExceptionHandler(LoaderType):
    class StreamLoaderExceptionCatcher(LoaderType):
        def run(self):
            try:
                super().run()
            except Exception:
                traceback.print_exc()
                self.data.put(None)
                self.data.shut_down()

    return StreamLoaderExceptionCatcher
