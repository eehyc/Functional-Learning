import torch
import t_io

endless_iterators={}

def FetchDeviceInputs(o_test_loader, target_test_batch):
    with torch.no_grad():

        if not o_test_loader in endless_iterators:
            endless_iterators[o_test_loader] = iter(o_test_loader)

        this_iterator = endless_iterators[o_test_loader]

        if not target_test_batch == 0:
            if target_test_batch < 0:
                target_test_batch = len(o_test_loader)
            test_inputs = []
            test_labels = []

            for batch_idx in range(target_test_batch):
                try:
                    inputs, targets = next(this_iterator)
                except StopIteration:
                    this_iterator = iter(o_test_loader)
                    endless_iterators[o_test_loader] = this_iterator
                    inputs, targets = next(this_iterator)


                test_inputs.append(inputs)
                test_labels.append(targets)

            test_inputs = torch.cat(test_inputs, dim=0)
            test_inputs = t_io.ImgConvertTorchAndNumpy(test_inputs)

            test_labels = torch.cat(test_labels, dim=0)

        else:
            test_inputs = None
            test_labels = None


        return test_inputs, test_labels
