class HeartbeatHandlerMixin():
    "Heartbeat notifications based on the application's logging activity"

    def __init__(self, beat_func, min_interval=1, **kw):
        """
        @param beat_func: calls this function when a heartbeat is due
        @param min_interval: minimum time interval between heartbeats
        """
        super(HeartbeatHandlerMixin, self).__init__(**kw)
        self.min_interval = min_interval
        self.last_beat = 0
        self.beat = beat_func
        self._emitting = False

    def emit(self, record):
        if self._emitting:
            # prevent reenterance
            return

        try:
            self._emitting = True
            if (record.created - self.last_beat) > self.min_interval:
                try:
                    log_message = self.format(record)
                except:  # noqa
                    log_message = "Log record formatting error (%s:#%s)" % (record.filename, record.lineno)
                self.beat(log_message=log_message, heartbeat=record.created)
                self.last_beat = record.created
        finally:
            self._emitting = False
