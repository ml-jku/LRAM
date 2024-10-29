def make_tokenizer(kind, tokenizer_kwargs=None):
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}
    if kind == 'mulaw':
        from .mu_law_tokenizer import MuLawTokenizer
        return MuLawTokenizer(**tokenizer_kwargs)
    elif kind == 'minmax':
        from .minmax_tokenizer import MinMaxTokenizer
        return MinMaxTokenizer(**tokenizer_kwargs)
    elif kind == 'minmax2':
        from .minmax_tokenizer import MinMaxTokenizer2
        return MinMaxTokenizer2(**tokenizer_kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type {kind}")
