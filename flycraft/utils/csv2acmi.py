import pandas as pd
from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent

def csv2acmi(
    traj_df: pd.DataFrame, 
    plane_id: str="A0100",
    time_column_name: str="time",
    lon_column_name: str="Longitude", 
    lat_column_name: str="Latitude", 
    alt_column_name: str="Altitude", 
    roll_column_name: str="Roll", 
    pitch_column_name: str="Pitch", 
    yaw_column_name: str="Yaw",
    acmi_file_save_dir: Path=PROJECT_ROOT_DIR / "expert_traj" / "acmi",
    acmi_file_name: str="test.acmi",
):
    assert time_column_name in traj_df.columns, f"DataFrame中不包含{time_column_name}列！！！"
    assert lon_column_name in traj_df.columns, f"DataFrame中不包含{lon_column_name}列！！！"
    assert lat_column_name in traj_df.columns, f"DataFrame中不包含{lat_column_name}列！！！"
    assert alt_column_name in traj_df.columns, f"DataFrame中不包含{alt_column_name}列！！！"
    assert roll_column_name in traj_df.columns, f"DataFrame中不包含{roll_column_name}列！！！"
    assert pitch_column_name in traj_df.columns, f"DataFrame中不包含{pitch_column_name}列！！！"
    assert yaw_column_name in traj_df.columns, f"DataFrame中不包含{yaw_column_name}列！！！"

    acmi_file = acmi_file_save_dir / acmi_file_name

    with open(acmi_file, mode='w', encoding='utf-8-sig') as f:
            f.write("FileType=text/acmi/tacview\n")
            f.write("FileVersion=2.1\n")
            f.write("0,ReferenceTime=2023-05-01T00:00:00Z\n")

    with open(acmi_file, mode='a', encoding='utf-8-sig') as f:
        for index, row in traj_df.iterrows():
            f.write(f"#{row[time_column_name]:.2f}\n")
            out_str = f"{plane_id},T={row[lon_column_name]}|{row[lat_column_name]}|{row[alt_column_name]}|{row[roll_column_name]}|{row[pitch_column_name]}|{row[yaw_column_name]},Name=F16,Color=Red\n"
            f.write(out_str)