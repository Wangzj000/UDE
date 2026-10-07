def print_args(args):
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"Task Name:":<20}{args.task_name:<20}{"Is Training:":<20}{args.is_training:<20}')
    print(f'  {"Model ID:":<20}{args.model_id:<20}{"Model:":<20}{args.model:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    print(f'  {"Checkpoints:":<20}{args.checkpoints:<20}')
    print()

    if args.task_name in ['long_term_forecast', 'short_term_forecast']:
        print("\033[1m" + "Forecasting Task" + "\033[0m")
        print(f'  {"Seq Len:":<20}{args.seq_len:<20}{"Pred Len:":<20}{args.pred_len:<20}')
        print()


    print("\033[1m" + "Model Parameters" + "\033[0m")
    print(f'  {"Variate Num:":<20}{args.n_vars:<20}{"d model:":<20}{args.d_model:<20}')
    print(f'  {"n heads:":<20}{args.n_heads:<20}{"e layers:":<20}{args.e_layers:<20}')
    print(f'  {"d layers:":<20}{args.d_layers:<20}{"d FF:":<20}{args.d_ff:<20}')
    print(f'  {"Dropout:":<20}{args.dropout:<20}{"Activation:":<20}{args.activation:<20}')
    print(f'  {"Output Attention:":<20}{args.output_attention:<20}')
    print()

    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Itr:":<20}{args.itr:<20}')
    print(f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    print(f'  {"Patience:":<20}{args.patience:<20}{"Learning Rate:":<20}{args.learning_rate:<20}')
    print(f'  {"Des:":<20}{args.des:<20}{"Loss:":<20}{args.loss:<20}')
    print(f'  {"Lradj:":<20}{args.lradj:<20}{"Use Amp:":<20}{args.use_amp:<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    print(f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')
    print()
    
    print("\033[1m" + "Delayformer" + "\033[0m")
    print(f'  {"Embedding Dim:":<20}{args.L:<20}')
    print(f'  {"p1:":<20}{args.p1:<20}{"p2:":<20}{args.p2:<20}')
    print()
    if args.task_name == "pretrain":
        print("\033[1m" + "Datasets" + "\033[0m")
        print(f'  {"Train datasets:":<20}{args.train_dataset_list:<20}')
        print(f'  {"Vali datasets:":<20}{args.train_dataset_list:<20}')
        print(f'  {"Test datasets:":<20}{args.test_dataset_list:<20}')
        print()
    else:
        print(f'  {"Train datasets:":<20}{args.data_path:<20}')
        print()