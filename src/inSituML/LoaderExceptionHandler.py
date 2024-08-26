import traceback


def wrapLoaderWithExceptionHandler(LoaderType, trainBF):
    class StreamLoaderExceptionCatcher(LoaderType):
        def run(self):
            try:
                super().run()
            except Exception:
                traceback.print_exc()
                self.data.put(None)
                trainBF.openpmdProduction=False

    return StreamLoaderExceptionCatcher
