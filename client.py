class Client:
    def __init__(self, idx, args, malicious, init_model, train_loader, global_test_loader):
        self.idx = idx
        self.args = args
        self.malicious = malicious
        self.train_loader = train_loader
        self.global_test_loader = global_test_loader

        # CELL
        self.eita_hat = self.args.eita
        self.eita = self.eita_hat
        self.alpha = self.args.alpha
        self.num_data = len(self.train_loader)

        self.init_model = init_model
        self.local_model = None
        self.global_model = None
        
