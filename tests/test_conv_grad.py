"""
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
from torch import nn
import models
from torch.autograd import grad
import numpy as np


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    t = torch.float64
    model = models.ClassifierLarger2([3, 32, 32]).to("cuda").to(t)
    img = nn.Parameter(torch.randn(128, 3, 32, 32, device='cuda')).to(t)

    params = list(model.parameters())
    y = model(img)
    dy = torch.rand_like(y)
    dp = grad(y, params, grad_outputs=dy, create_graph=True)
    ddp = [torch.rand_like(p) for p in params]
    dimg, = grad(dp, img, grad_outputs=ddp)

    # Check that we get the same gradients when running it twice
    img2 = nn.Parameter(img + 0)

    params = list(model.parameters())
    y2 = model(img2)
    dp2 = grad(y2, params, grad_outputs=dy, create_graph=True)
    dimg2, = grad(dp2, img2, grad_outputs=ddp)
    assert np.allclose(dimg.detach().cpu(), dimg2.detach().cpu())

    for module in model.modules():
        module.disable_exconv = True
    img2 = nn.Parameter(img + 0)

    params = list(model.parameters())
    y2 = model(img2)
    assert np.allclose(y.detach().cpu(), y2.detach().cpu())
    dp2 = grad(y2, params, grad_outputs=dy, create_graph=True)
    import pdb; pdb.set_trace()
    for a, b in zip(dp, dp2):
        assert np.allclose(a.detach().cpu(), b.detach().cpu())
    dimg2, = grad(dp2, img2, grad_outputs=ddp)
    import pdb; pdb.set_trace()
    assert np.allclose(dimg.detach().cpu(), dimg2.detach().cpu())


if __name__ == "__main__":
    main()
