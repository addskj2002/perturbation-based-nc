

import os

import torch


"""
Here we implement the projection from any arbitrary vector in R^d to a subspace spanned by
some collection of vectors in R^d. Here, `subspace` is a 2D torch tensor G with each row
being one of the spanning vectors; and `vectors` is a 2D torch tensor V with each row
being one of the vectors to be projected. The desired output can be formulated as

    V_proj = VG^T (GG^T)^{-1} G

`project_part_1` computes GV^T
`compute_GGT_inv` computes (GG^T)^{-1}
`project_part_2` computes G^T (GG^T)^{-1} GV^T using the results from the previous steps
`project` combines the three steps above to compute the final output V_proj
"""


def project_part_1(vectors, subspace, device):
    # If subspace is a directory, load from there and do patch-by-patch multiplication
    if isinstance(subspace, str):
        idx = 0
        ret = []
        while os.path.exists(f"{subspace}/{idx}.pth"):
            ret.append(
                torch.load(f"{subspace}/{idx}.pth", weights_only=False, map_location=device)
                @ vectors.T
            )
            idx += 1
        return torch.cat(ret)

    # Otherwise just use regular matrix multiplication
    else:
        return subspace @ vectors.T
    

def compute_GGT_inv(subspace, device, overwrite=True, cache=None):
    # Use cache
    if not overwrite and (
        cache is not None and os.path.exists(cache)
    ):
        filedata = torch.load(cache, weights_only=False)
        if "GGT-1" in filedata:
            return filedata["GGT-1"]

    # GGT using patch-by-patch computation for large subspaces
    if isinstance(subspace, str):
        idx = 0
        ret = []
        while os.path.exists(f"{subspace}/{idx}.pth"):
            jdx = 0
            rows = []
            while os.path.exists(f"{subspace}/{jdx}.pth"):
                rows.append(
                    torch.load(f"{subspace}/{idx}.pth", weights_only=False, map_location=device)
                    @ torch.load(f"{subspace}/{jdx}.pth", weights_only=False, map_location=device).T
                )
                jdx += 1
            ret.append(torch.cat(rows, dim=1))
            idx += 1
        ggt = torch.cat(ret)

    # GGT using regular matrix multiplication
    else:
        ggt = subspace @ subspace.T

    # Pseudo-inverse
    ggt_inverse = torch.pinverse(ggt)
    
    # Caching part 2
    if cache is not None:
        filedata = {}
        if os.path.exists(cache):
            filedata = torch.load(cache, weights_only=False)
        filedata["GGT-1"] = ggt_inverse
        torch.save(filedata, cache)

    return ggt_inverse


def project_part_2(subspace, ggt_inverse, intermediate, device):
    intermediate = ggt_inverse @ intermediate

    # If subspace is a directory, load from there and do patch-by-patch multiplication
    if isinstance(subspace, str):
        idx = 0
        jdx = 0
        ret = None
        while os.path.exists(f"{subspace}/{idx}.pth"):
            partial_subspace = torch.load(
                f"{subspace}/{idx}.pth", weights_only=False, map_location=device
            )
            N, _ = partial_subspace.shape
            if ret is None:
                ret = partial_subspace.T @ intermediate[jdx:jdx+N]
            else:
                ret += partial_subspace.T @ intermediate[jdx:jdx+N]
            jdx += N
            idx += 1
        return ret

    # Otherwise just use regular matrix multiplication
    else:
        return subspace.T @ intermediate


def project(vectors, subspace, device, overwrite=True, cache=None):
    intermediate = project_part_1(vectors, subspace, device)
    ggt_inverse = compute_GGT_inv(subspace, device, overwrite, cache)
    ret = project_part_2(subspace, ggt_inverse, intermediate, device).T
    return ret


if __name__ == "__main__":
    pass
