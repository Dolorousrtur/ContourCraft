import torch


def get_face_areas(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    u = v2 - v0
    v = v1 - v0

    if u.shape[-1] == 2:
        out = torch.abs(torch.cross(u, v)) / 2.0
    else:
        out = torch.linalg.norm(torch.cross(u, v), axis=-1) / 2.0
    return out


def get_vertex_mass(vertices, faces, density):
    '''
    Computes the mass of each vertex according to triangle areas and fabric density
    '''
    areas = get_face_areas(vertices, faces)
    triangle_masses = density * areas

    vertex_masses = torch.zeros(vertices.shape[0]).to(vertices.device)

    vertex_masses = torch.index_add(vertex_masses, 0, faces[:, 0], triangle_masses / 3)
    vertex_masses = torch.index_add(vertex_masses, 0, faces[:, 1], triangle_masses / 3)
    vertex_masses = torch.index_add(vertex_masses, 0, faces[:, 2], triangle_masses / 3)

    return vertex_masses