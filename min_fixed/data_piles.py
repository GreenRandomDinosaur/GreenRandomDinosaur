import torchvision.transforms as transforms
import torchvision.datasets as datasets


def prepare_dataset(args, train_set=True, tiny_dir=''):
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

        if train_set: 
            dataset = datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform_train)
        else: 
            dataset = datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
        num_classes = 10


    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
       
        if train_set: 
            dataset = datasets.CIFAR100(root='../../data', train=True, download=True, transform=transform_train)
        else: 
            dataset = datasets.CIFAR100(root='../../data', train=False, download=True, transform=transform_test)
        num_classes = 100
       
        
    elif args.dataset == 'mnist':
        transform_train=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        transform_test=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        if train_set: 
            dataset = datasets.MNIST(root='../../data', train=True, download=True, transform=transform_train)
        else: 
            dataset = datasets.MNIST(root='../../data', train=False, download=True, transform=transform_test)
        num_classes = 10
        

    elif args.dataset == 'tinyimagenet':
        tiny_mean = [0.48024578664982126, 0.44807218089384643, 0.3975477478649648]
        tiny_std = [0.2769864069088257, 0.26906448510256, 0.282081906210584]
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])

        if train_set: 
            dataset = datasets.ImageFolder(
                tiny_dir+'/train',
                transform_train)
        else: 
            dataset = datasets.ImageFolder(
                tiny_dir+'/test',
                transform_test)        
            
        num_classes = 200
        
        
    return dataset, num_classes