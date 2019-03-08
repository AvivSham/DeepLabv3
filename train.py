import torch
import torch.nn as nn
from utils import *
from models.deeplabv3 import DeepLabv3
import sys
from tqdm import tqdm

def train(FLAGS):

    # Defining the hyperparameters
    device =  FLAGS.cuda
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    lr = FLAGS.learning_rate
    print_every = FLAGS.print_every
    eval_every = FLAGS.eval_every
    save_every = FLAGS.save_every
    nc = FLAGS.num_classes
    wd = FLAGS.weight_decay

    ip = FLAGS.input_path_train
    lp = FLAGS.label_path_train

    ipv = FLAGS.input_path_val
    lpv = FLAGS.label_path_val
    
    H = FLAGS.resize_height
    W = FLAGS.resize_width

    dtype = FLAGS.dtype
    sched = FLAGS.scheduler
    
    if FLAGS.dtype == 'cityscapes':
        train_samples = len(glob.glob(ip + '/**/*.png', recursive=True))
        eval_samples = len(glob.glob(lp + '/**/*.png', recursive=True))
    elif FLAGS.dtype == 'pascal':
        train_samples = len(os.listdir(lp))
        eval_samples = len(os.listdir(lp))

    print ('[INFO]Defined all the hyperparameters successfully!')
    
    # Get the class weights
    #print ('[INFO]Starting to define the class weights...')
    #pipe = loader(ip, lp, batch_size='all')
    #class_weights = get_class_weights(pipe, nc)
    #print ('[INFO]Fetched all class weights successfully!')

    # Get an instance of the model
    model = DeepLabv3(nc)
    print ('[INFO]Model Instantiated!')
    
    # Move the model to cuda if available
    model.to(device)

    # Define the criterion and the optimizer
    #criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=wd)
    print ('[INFO]Defined the loss function and the optimizer')

    # Training Loop starts
    print ('[INFO]Staring Training...')
    print ()

    train_losses = []
    eval_losses = []
    
    if dtype == 'cityscapes':
        pipe = loader_cscapes(ip, lp, batch_size, h = H, w = W)
    elif dtype == 'pascal':
        pipe = loader(ip, lp, batch_size, h = H, w = W)
    #eval_pipe = loader(ipv, lpv, batch_size)

    show_every = 250

    train_losses = []
    eval_losses = []

    bc_train = train_samples // batch_size
    bc_eval = eval_samples // batch_size

    if sched:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (1 - (epoch / epochs)) ** 0.9)

    for e in range(1, epochs+1):

        train_loss = 0
        print ('-'*15,'Epoch %d' % e, '-'*15)
        
        if sched:
            scheduler.step()
        
        model.train()

        for ii in tqdm(range(bc_train)):
            X_batch, mask_batch = next(pipe)
            
            X_batch, mask_batch = X_batch.to(device), mask_batch.to(device)

            optimizer.zero_grad()

            out = model(X_batch.float())
            
            loss = criterion(out, mask_batch.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            if ii % show_every == 0:
                out5 = show_cscpaes(model, H, W)
                checkpoint = {
                    'epochs' : e,
                    'model_state_dict' : model.state_dict(),
                    'opt_state_dict' : optimizer.state_dict()
                }
                torch.save(checkpoint, './ckpt-dlabv3-{}-{:2f}.pth'.format(e, train_loss))
                print ('Model saved!')

        print ()
        train_losses.append(train_loss)

        if (e+1) % print_every == 0:
            print ('Epoch {}/{}...'.format(e, epochs),
                    'Loss {:6f}'.format(train_loss))
        
        if e % save_every == 0:

            show_pascal(model, training_path, all_tests[np.random.randint(0, len(all_tests))])
            checkpoint = {
                'epochs' : e,
                'state_dict' : model.state_dict()
            }
            torch.save(checkpoint, '/content/ckpt-enet-{}-{:2f}.pth'.format(e, train_loss))
            print ('Model saved!')
        
        
    #     show(model, all_tests[np.random.randint(0, len(all_tests))])
    #     show_pascal(model, training_path, all_tests[np.random.randint(0, len(all_tests))])

    print ('[INFO]Training Process complete!')
