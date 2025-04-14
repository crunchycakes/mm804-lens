import os
import tempfile
import vtk
from flask import Flask, request, send_file
import numpy as np

app = Flask(__name__)

# ----------------------------
# VTK Core Processing Functions
# ----------------------------
def read_obj(file_name):
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File {file_name} not found!")
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
        print("Smoothing filter failed, returning original data:", e)
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

# ----------------------------
# New: Noise and outlier removal (adjusted strategy to preserve more original topology)
# ----------------------------
def remove_noise_outliers(polydata, area_threshold=50.0):
    """
    Uses vtkPolyDataConnectivityFilter to extract all regions,
    keeping only those larger than threshold to preserve real details.
    """
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

# ----------------------------
# New: Mesh reconstruction (using surface reconstruction and contour extraction)
# ----------------------------
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

# ----------------------------
# New: Multi-scale processing (combining results from different smoothing levels)
# ----------------------------
def multi_scale_processing(polydata):
    # Light smoothing
    smooth1 = advanced_smooth_mesh(polydata, iterations=1, pass_band=0.8, feature_angle=80.0)
    # Stronger smoothing
    smooth2 = advanced_smooth_mesh(polydata, iterations=2, pass_band=0.7, feature_angle=75.0)
    
    # Combine both results
    appendFilter = vtk.vtkAppendPolyData()
    appendFilter.AddInputData(smooth1)
    appendFilter.AddInputData(smooth2)
    appendFilter.Update()
    
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(appendFilter.GetOutput())
    clean.Update()
    return clean.GetOutput()

# ----------------------------
# New: Topology preservation
# ----------------------------
def preserve_topology(original, processed):
    """
    Extracts feature edges from original mesh and merges them with processed mesh
    to maintain original topological structure.
    """
    # Extract feature edges from original
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

# ----------------------------
# Complete processing pipeline (preserving original topology)
# ----------------------------
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
# Flask API Endpoints
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

        # Print optimization info
        orig_points = vtk_mesh.GetNumberOfPoints()
        orig_polys = vtk_mesh.GetNumberOfPolys()
        proc_points = processed_mesh.GetNumberOfPoints()
        proc_polys = processed_mesh.GetNumberOfPolys()
        print("Optimization Info (API Processing):")
        print("Original model: vertices =", orig_points, "faces =", orig_polys)
        print("Processed model: vertices =", proc_points, "faces =", proc_polys)
        if orig_points:
            print("Vertex reduction: {:.2f}%".format(100 * (orig_points - proc_points) / orig_points))
        if orig_polys:
            print("Face reduction: {:.2f}%".format(100 * (orig_polys - proc_polys) / orig_polys))

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".obj").name
        write_obj(processed_mesh, output_path)
    except Exception as e:
        return f"Processing error: {e}", 500

    return send_file(output_path, as_attachment=True, download_name="processed.obj")

# ----------------------------
# Main Program Entry
# ----------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Mesh Processing with Topology Preservation using VTK and Flask")
    parser.add_argument('--input', type=str, help="Input OBJ file path")
    parser.add_argument('--output', type=str, help="Output OBJ file path")
    parser.add_argument('--runserver', action='store_true', help="Run as Flask API server")
    parser.add_argument('--showcurv', action='store_true', help="Visualize input model curvature")
    args = parser.parse_args()

    if args.runserver:
        app.run(port=5000, debug=True)
    elif args.input and args.output:
        try:
            vtk_mesh = read_obj(args.input)
            print("Original model: vertices =", vtk_mesh.GetNumberOfPoints(), "faces =", vtk_mesh.GetNumberOfPolys())
            
            if args.showcurv:
                mesh_curv = compute_curvature(vtk_mesh)
                visualize_curvature(mesh_curv)
            
            processed_mesh = process_full_pipeline(vtk_mesh)
            write_obj(processed_mesh, args.output)
            print(f"Processed model saved to: {args.output}")

            # Print optimization statistics
            orig_points = vtk_mesh.GetNumberOfPoints()
            orig_polys = vtk_mesh.GetNumberOfPolys()
            proc_points = processed_mesh.GetNumberOfPoints()
            proc_polys = processed_mesh.GetNumberOfPolys()
            print("Optimization Info:")
            print("Original model: vertices =", orig_points, "faces =", orig_polys)
            print("Processed model: vertices =", proc_points, "faces =", proc_polys)
            if orig_points:
                print("Vertex reduction: {:.2f}%".format(100 * (orig_points - proc_points) / orig_points))
            if orig_polys:
                print("Face reduction: {:.2f}%".format(100 * (orig_polys - proc_polys) / orig_polys))
            
            visualize_comparison(vtk_mesh, processed_mesh)
        except Exception as e:
            print("Processing error:", e)
    else:
        print("Please provide --input and --output parameters, or use --runserver to start API.")
