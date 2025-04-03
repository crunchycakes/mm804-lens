import os
import tempfile
import vtk
from flask import Flask, request, send_file

app = Flask(__name__)

def read_obj(file_name):
    
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"文件 {file_name} 不存在！")
    reader = vtk.vtkOBJReader()
    reader.SetFileName(file_name)
    reader.Update()
    return reader.GetOutput()

def repair_mesh(polydata, hole_size=100.0):
   
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(polydata)
    cleaner.Update()
    cleaned = cleaner.GetOutput()

    fillHoles = vtk.vtkFillHolesFilter()
    fillHoles.SetInputData(cleaned)
    fillHoles.SetHoleSize(hole_size)
    fillHoles.Update()
    return fillHoles.GetOutput()

def advanced_smooth_mesh(polydata, iterations=3, pass_band=0.5, feature_angle=60.0):
    
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(polydata)
    smoother.SetNumberOfIterations(iterations)
    smoother.BoundarySmoothingOff()
    smoother.SetPassBand(pass_band)
    smoother.SetFeatureAngle(feature_angle)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return smoother.GetOutput()

def decimate_mesh(polydata, reduction=0.05):
    
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(polydata)
    decimate.SetTargetReduction(reduction)
    decimate.PreserveTopologyOn()
    decimate.Update()
    return decimate.GetOutput()

def deform_mesh(polydata, factor=1.0):
   
    transform = vtk.vtkTransform()
   
    transform.Scale(1.0 + factor * 0.1, 1.0 + factor * 0.1, 1.0 + factor * 0.1)
    
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(polydata)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    return transform_filter.GetOutput()

def recalc_normals(polydata):
    
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOn()
    normals.Update()
    return normals.GetOutput()

def compute_curvature(polydata):
   
    curvature_filter = vtk.vtkCurvatures()
    curvature_filter.SetInputData(polydata)
    curvature_filter.SetCurvatureTypeToMean()
    curvature_filter.Update()
    return curvature_filter.GetOutput()

def visualize_curvature(polydata):
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetScalarRange(polydata.GetPointData().GetScalars().GetRange())
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(800, 600)

    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.4)

    renderWindow.Render()
    renderWindowInteractor.Start()

def write_obj(polydata, file_name):
   
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(file_name)
    writer.SetInputData(polydata)
    writer.Update()  # 调用 Update() 后再 Write() 确保数据正确写入
    writer.Write()

def visualize_comparison(original, processed):
   
  
    mapper1 = vtk.vtkPolyDataMapper()
    mapper1.SetInputData(original)
    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper1)

    
    mapper2 = vtk.vtkPolyDataMapper()
    mapper2.SetInputData(processed)
    actor2 = vtk.vtkActor()
    actor2.SetMapper(mapper2)

   
    renderer1 = vtk.vtkRenderer()
    renderer2 = vtk.vtkRenderer()

    
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(1200, 600)
    renderWindow.AddRenderer(renderer1)
    renderWindow.AddRenderer(renderer2)

   
    renderer1.SetViewport(0.0, 0.0, 0.5, 1.0)
    renderer2.SetViewport(0.5, 0.0, 1.0, 1.0)

    renderer1.AddActor(actor1)
    renderer2.AddActor(actor2)

    renderer1.SetBackground(0.1, 0.2, 0.4)
    renderer2.SetBackground(0.1, 0.2, 0.4)

    
    text_actor1 = vtk.vtkTextActor()
    text_actor1.SetInput("Original Mesh")
    text_actor1.GetTextProperty().SetFontSize(24)
    text_actor1.GetTextProperty().SetColor(1.0, 1.0, 1.0)
    text_actor1.SetDisplayPosition(10, 10)
    renderer1.AddActor2D(text_actor1)

    text_actor2 = vtk.vtkTextActor()
    text_actor2.SetInput("Processed Mesh")
    text_actor2.GetTextProperty().SetFontSize(24)
    text_actor2.GetTextProperty().SetColor(1.0, 1.0, 1.0)
    text_actor2.SetDisplayPosition(10, 10)
    renderer2.AddActor2D(text_actor2)

   
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindow.Render()
    renderWindowInteractor.Start()

@app.route('/process', methods=['POST'])
def process_mesh_api():
    """
    Flask API 接口：上传 OBJ 文件，返回处理后的 OBJ 文件。
    前端需通过字段名 "mesh" 上传 OBJ 文件。
    """
    if 'mesh' not in request.files:
        return "No file part", 400

    file = request.files['mesh']
    if file.filename == '':
        return "No selected file", 400

   
    input_path = tempfile.NamedTemporaryFile(delete=False, suffix=".obj").name
    file.save(input_path)

    try:
        # 处理流程：读取、修复、平滑、简化、变形、重新计算法线
        mesh = read_obj(input_path)
        mesh_repaired = repair_mesh(mesh, hole_size=100.0)
        mesh_smoothed = advanced_smooth_mesh(mesh_repaired, iterations=3, pass_band=0.5, feature_angle=60.0)
        mesh_decimated = decimate_mesh(mesh_smoothed, reduction=0.05)
        mesh_deformed = deform_mesh(mesh_decimated, factor=1.0)
        mesh_final = recalc_normals(mesh_deformed)
        

        
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".obj").name
        write_obj(mesh_final, output_path)
    except Exception as e:
        return f"Error during processing: {e}", 500

    return send_file(output_path, as_attachment=True, download_name="processed.obj")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Mesh Processing using VTK and Flask")
    parser.add_argument('--input', type=str, help="输入 OBJ 文件路径")
    parser.add_argument('--output', type=str, help="输出 OBJ 文件路径")
    parser.add_argument('--runserver', action='store_true', help="以 Flask API 模式运行")
    parser.add_argument('--showcurv', action='store_true', help="显示输入模型的曲率分布")
    args = parser.parse_args()

    if args.runserver:
        # 启动 API 服务
        app.run(port=5000, debug=True)
    elif args.input and args.output:
        try:
            mesh = read_obj(args.input)
            print("原始模型：顶点数 =", mesh.GetNumberOfPoints(), "面数 =", mesh.GetNumberOfPolys())
            
           
            if args.showcurv:
                mesh_curv = compute_curvature(mesh)
                visualize_curvature(mesh_curv)
            
            mesh_repaired = repair_mesh(mesh, hole_size=100.0)
            mesh_smoothed = advanced_smooth_mesh(mesh_repaired, iterations=3, pass_band=0.5, feature_angle=60.0)
            mesh_decimated = decimate_mesh(mesh_smoothed, reduction=0.05)
            mesh_deformed = deform_mesh(mesh_decimated, factor=1.0)
            mesh_final = recalc_normals(mesh_deformed)
            
            write_obj(mesh_final, args.output)
            print(f"处理后的模型已保存为: {args.output}")
            
            visualize_comparison(mesh, mesh_final)
        except Exception as e:
            print("处理过程中发生错误：", e)
    else:
        print("请提供 --input 和 --output 参数，或使用 --runserver 启动 API 服务。")
