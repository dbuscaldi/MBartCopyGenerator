# MBartCopyGenerator
MBart-based model for the indexing of scientific documents

Coded by https://gist.github.com/jogonba2

Presented to NLDB 2022

"Transformer-based models for the Automatic Indexing of Scientific Documents in French"
José Angel Gonzalez, Davide Buscaldi, Lluis Hurtado and Emilio Sanchis


You can use this class in run_summarization.py, adding some arguments to the parser, and calling the MBartCopyGenerator when loading the model.
```
if model_args.copy_enhanced:
        logger.info("Using a copy enhanced version of MBart")
        model_type = MBartCopyGenerator
        config.update({"centrality": False, "tf_idf": False}) # update the config if needed.
    else:
        model_type = AutoModelForSeq2SeqLM
    model = model_type.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
```
