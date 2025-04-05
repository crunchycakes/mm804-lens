import os
import tempfile
import vtk
from flask import Flask, request, send_file
import numpy as np

app = Flask(__name__)

# ----------------------------
# VTK 基础处理函数
# ----------------------------
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

def advanced_smooth_mesh(polydata, iterations=1, pass_band=0.8, feature_angle=80.0):
    try:
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
    except Exception as e:
        print("平滑滤波器出错，降级为直接返回原始数据:", e)
        return polydata

def decimate_mesh(polydata, reduction=0.01):
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

def write_obj(polydata, file_name):
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(file_name)
    writer.SetInputData(polydata)
    writer.Update()  
    writer.Write()

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


def remove_noise_outliers(polydata, area_threshold=50.0):
   
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(polydata)
    connectivity.SetExtractionModeToAllRegions()  
    connectivity.Update()
    
    num_regions = connectivity.GetNumberOfExtractedRegions()
    appendFilter = vtk.vtkAppendPolyData()
    for i in range(num_regions):
        connectivity.SetExtractionModeToSpecifiedRegions()
        connectivity.InitializeSpecifiedRegionList()
        connectivity.AddSpecifiedRegion(i)
        connectivity.Update()
        region = connectivity.GetOutput()
        if region.GetNumberOfPoints() > area_threshold:
            appendFilter.AddInputData(region)
    appendFilter.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(appendFilter.GetOutput())
    clean.Update()
    return clean.GetOutput()


def reconstruct_mesh(polydata):
    surfaceReconstruction = vtk.vtkSurfaceReconstructionFilter()
    surfaceReconstruction.SetInputData(polydata)
    surfaceReconstruction.Update()
    
    contourFilter = vtk.vtkContourFilter()
    contourFilter.SetInputConnection(surfaceReconstruction.GetOutputPort())
    contourFilter.SetValue(0, 0.0)
    contourFilter.Update()
    
    reverseSense = vtk.vtkReverseSense()
    reverseSense.SetInputConnection(contourFilter.GetOutputPort())
    reverseSense.ReverseCellsOn()
    reverseSense.ReverseNormalsOn()
    reverseSense.Update()
    
    return reverseSense.GetOutput()


def multi_scale_processing(polydata):
    smooth1 = advanced_smooth_mesh(polydata, iterations=1, pass_band=0.8, feature_angle=80.0)
    smooth2 = advanced_smooth_mesh(polydata, iterations=2, pass_band=0.7, feature_angle=75.0)
    
    appendFilter = vtk.vtkAppendPolyData()
    appendFilter.AddInputData(smooth1)
    appendFilter.AddInputData(smooth2)
    appendFilter.Update()
    
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(appendFilter.GetOutput())
    clean.Update()
    return clean.GetOutput()


def preserve_topology(original, processed):

    featureEdges = vtk.vtkFeatureEdges()
    featureEdges.SetInputData(original)
    featureEdges.BoundaryEdgesOn()
    featureEdges.FeatureEdgesOn()    
    featureEdges.ManifoldEdgesOff()
    featureEdges.NonManifoldEdgesOff()
    featureEdges.Update()
    boundaries = featureEdges.GetOutput()
    
    appendFilter = vtk.vtkAppendPolyData()
    appendFilter.AddInputData(processed)
    appendFilter.AddInputData(boundaries)
    appendFilter.Update()
    
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(appendFilter.GetOutput())
    clean.Update()
    return clean.GetOutput()


def process_full_pipeline(vtk_mesh):
    mesh_repaired = repair_mesh(vtk_mesh, hole_size=100.0)
    mesh_smoothed = advanced_smooth_mesh(mesh_repaired, iterations=1, pass_band=0.8, feature_angle=80.0)
    mesh_decimated = decimate_mesh(mesh_smoothed, reduction=0.01)
    mesh_deformed = deform_mesh(mesh_decimated, factor=1.0)
    mesh_final = recalc_normals(mesh_deformed)
    
    mesh_clean = remove_noise_outliers(mesh_final, area_threshold=50.0)
    
    mesh_multi = multi_scale_processing(mesh_clean)
    
    mesh_recon = reconstruct_mesh(mesh_multi)
    
    mesh_preserved = preserve_topology(vtk_mesh, mesh_recon)
    
    return mesh_preserved

# ----------------------------
# Flask API 
# ----------------------------
@app.route('/process', methods=['POST'])
def process_mesh_api():
    if 'mesh' not in request.files:
        return "No file part", 400

    file = request.files['mesh']
    if file.filename == '':
        return "No selected file", 400

    input_path = tempfile.NamedTemporaryFile(delete=False, suffix=".obj").name
    file.save(input_path)

    try:
        vtk_mesh = read_obj(input_path)
        processed_mesh = process_full_pipeline(vtk_mesh)
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".obj").name
        write_obj(processed_mesh, output_path)
    except Exception as e:
        return f"Error during processing: {e}", 500

    return send_file(output_path, as_attachment=True, download_name="processed.obj")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Mesh Processing with Topology Preservation using VTK and Flask")
    parser.add_argument('--input', type=str, help="输入 OBJ 文件路径")
    parser.add_argument('--output', type=str, help="输出 OBJ 文件路径")
    parser.add_argument('--runserver', action='store_true', help="以 Flask API 模式运行")
    parser.add_argument('--showcurv', action='store_true', help="显示输入模型的曲率分布")
    args = parser.parse_args()

    if args.runserver:
        app.run(port=5000, debug=True)
    elif args.input and args.output:
        try:
            vtk_mesh = read_obj(args.input)
            print("原始模型：顶点数 =", vtk_mesh.GetNumberOfPoints(), "面数 =", vtk_mesh.GetNumberOfPolys())
            
            if args.showcurv:
                mesh_curv = compute_curvature(vtk_mesh)
                visualize_curvature(mesh_curv)
            
            processed_mesh = process_full_pipeline(vtk_mesh)
            write_obj(processed_mesh, args.output)
            print(f"处理后的模型已保存为: {args.output}")
            
            visualize_comparison(vtk_mesh, processed_mesh)
        except Exception as e:
            print("处理过程中发生错误：", e)
    else:
        print("请提供 --input 和 --output 参数，或使用 --runserver 启动 API 服务。")
