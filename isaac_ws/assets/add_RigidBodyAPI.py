import os
from pxr import Usd, UsdPhysics, UsdGeom, Gf

def add_rigid_body_api(usd_file_path):
    # Open the USD stage
    stage = Usd.Stage.Open(usd_file_path)

    # Get the default prim
    default_prim = stage.GetDefaultPrim()

    # Check if default prim exists and if RigidBodyAPI already exists
    if default_prim and not default_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        # Add RigidBodyAPI
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(default_prim)
        if rigid_body_api:
            print(f"Added RigidBodyAPI to {usd_file_path}")
            # Save the changes
            stage.GetRootLayer().Save()
    
    def convert_properties(prim):
        for prop_name, prop in prim.GetProperties():
            # Check if the property is named "orient" and if it is a quaternion
            if prop_name == "orient" and prop.GetTypeName() == "Quatd":
                # Get the current value
                value = prop.Get()

                # Convert the value to GfQuatf
                value_f = Gf.Quatf(value)

                # Set the converted value
                prop.Set(value_f)

        # Recursively process children
        for child in prim.GetChildren():
            convert_properties(child)
            
    convert_properties(root_prim)
    
    UsdPhysics.CollisionAPI.Apply(default_prim)
    stage.Save()
    

# Path to the folder containing USD files
folder_path = os.getcwd()

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if not filename.endswith(".py"):
        filename += "/meshes/_converted/model_obj.usd"
    if filename.endswith(".usd"):
        usd_file_path = os.path.join(folder_path, filename)
        add_rigid_body_api(usd_file_path)
