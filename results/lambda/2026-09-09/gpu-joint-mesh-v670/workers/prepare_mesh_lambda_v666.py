from pathlib import Path
p=Path('build/performance')
text=(p/'prepare_geometry_lambda_v655.py').read_text()
text=text.replace('spacepdhcg-joint-geometry-v651','spacepdhcg-joint-mesh-v662').replace('spacepdhcg-joint-geometry-v655','spacepdhcg-joint-mesh-v666').replace('joint-geometry-v655.tar.gz','joint-mesh-v666.tar.gz')
text=text.replace("validation=(root/'validation-runner-v654.py').read_text()","validation=(p/'check_joint_mesh_v665.py').read_text()")
text=text.replace('run_joint_geometry_benchmark_v653.py','run_joint_mesh_benchmark_v664.py')
text=text.replace('benchmark_joint_selection_v634.py','benchmark_joint_mesh_v664.py')
text=text.replace('launch_geometry_lambda_v655.py','launch_mesh_lambda_v666.py')
(p/'upload_mesh_lambda_v666.py').write_text(text)
