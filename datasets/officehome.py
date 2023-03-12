import os.path as osp

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class OfficeHome_BBDA(DatasetBase):
    """Office-Home BBDA.

    Statistics:
        - Around 15,500 images.
        - 65 classes related to office and home objects.
        - 4 domains: Art, Clipart, Product, Real World.
        - URL: http://hemanthdv.org/OfficeHome-Dataset/.

    Reference:
        - Venkateswara et al. Deep Hashing Network for Unsupervised
        Domain Adaptation. CVPR 2017.
    """

    dataset_dir = "office_home"
    domains = ["art", "clipart", "product", "real_world"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.check_input_domains(cfg.DATASET.SOURCE_DOMAINS,
                                 cfg.DATASET.TARGET_DOMAINS)

        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS)
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS)
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS)

        super().__init__(train_x=train_x, train_u=train_u, test=test)

    def _read_data(self, input_domains):
        items = []
        for domain, dname in enumerate(input_domains):
            txt_file = osp.join(self.dataset_dir, 'image_list', dname + '.txt')
            with open(txt_file, "r") as f:
                for line in f.readlines():
                    path, label = line.split()
                    label = int(label)
                    class_name = path.split('/')[-2]
                    if not osp.isabs(path):
                        path = osp.join(self.dataset_dir, path)

                    item = Datum(impath=path,
                                 label=label,
                                 domain=domain,
                                 classname=class_name.lower())
                    items.append(item)

        return items
