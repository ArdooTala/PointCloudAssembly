import copy
import numpy as np
import open3d as o3d


def draw_registration_result(source, target, transformation, viz=None):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    o3d.visualization.draw_geometries([source_temp, target_temp])


source = o3d.io.read_point_cloud("cloud 1.ply")
target = o3d.io.read_point_cloud("cloud 2.ply")
print(source)
threshold = 2
trans_init = np.identity(4)

draw_registration_result(source, target, trans_init)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(source)
vis.add_geometry(target)
icp_iteration = 200
save_image = False

source_temp = copy.deepcopy(source)
target_temp = copy.deepcopy(target)
source_temp.paint_uniform_color([1, 0.706, 0])
target_temp.paint_uniform_color([0, 0.651, 0.929])

vis2 = o3d.visualization.Visualizer()
vis2.create_window()
vis2.add_geometry(source_temp)
vis2.add_geometry(target_temp)

print("Initial alignment")
evaluation = o3d.pipelines.registration.evaluate_registration(
    source, target, threshold)
print(evaluation)

print("Apply point-to-point ICP")

for i in range(icp_iteration):
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10))
    print('='*100)
    print(threshold)
    print(reg_p2p)
    # threshold *= 0.99

    source.transform(reg_p2p.transformation)
    vis.update_geometry(source)
    vis.poll_events()
    vis.update_renderer()

    source_temp.transform(reg_p2p.transformation)
    vis2.update_geometry(source_temp)
    vis2.poll_events()
    vis2.update_renderer()


print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result(source, target, reg_p2p.transformation)