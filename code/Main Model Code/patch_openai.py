try:
    from openai._base_client import SyncHttpxClientWrapper
    _original_init = SyncHttpxClientWrapper.__init__
    def _patched_init(self, **kwargs):
        if 'proxies' in kwargs:
            kwargs.pop('proxies')
        _original_init(self, **kwargs)
    SyncHttpxClientWrapper.__init__ = _patched_init
    print('patch_openai: Successfully patched SyncHttpxClientWrapper')
except Exception as e:
    print('patch_openai: Failed to patch SyncHttpxClientWrapper:', e) 