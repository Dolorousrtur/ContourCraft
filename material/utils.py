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


def init_matstack(config, modules, aux_modules, dataloader_modules):
    dataloader_ft = dataloader_modules['finetune'].create_dataloader(is_eval=True)

    material_stack = aux_modules['material_stack']

    print('dataloader_ft', len(dataloader_ft))
    material_stack.initialize(dataloader_ft)
    material_stack = material_stack.to('cuda:0')

    print('material_stack', material_stack.materials.keys())

    mstack_name = list(config.material_stack.keys())[0]
    optimizer_material = modules['material_stack'].create_optimizer(material_stack, config.material_stack[mstack_name].optimizer)

    aux_modules['optimizer_material'] = optimizer_material

    return aux_modules
