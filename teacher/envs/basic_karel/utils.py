import numpy as np


def symb_2_bitmap(curr_task):
    """
    Karel task representation. From symbolic to bitmap representation.
    Bitmap representation is:
    [walls, agent_location, agent_orientation, premarkers,
    post_agent_loc, post_agent_orientation, post markers]
    :param: curr_task, dictionary representation of task
    :return: bitmap, a binary numpy array representation of task
    """
    bitmap = np.zeros((5 * curr_task["gridsz_num_rows"] * curr_task["gridsz_num_cols"] + 2 * 4))

    # Walls
    for wall in curr_task["walls"]:
        bitmap[wall[0] * curr_task["gridsz_num_rows"] + wall[1]] = 1

    # Agent location
    bitmap[1 * curr_task["gridsz_num_rows"] * curr_task["gridsz_num_cols"] +
           int(curr_task["pregrid_agent_row"]) * curr_task["gridsz_num_rows"] +
           int(curr_task["pregrid_agent_col"])] = 1

    # Agent Orientation
    if curr_task["pregrid_agent_dir"] == "north":
        bitmap[2 * curr_task["gridsz_num_rows"] * curr_task["gridsz_num_cols"] + 0] = 1
    elif curr_task["pregrid_agent_dir"] == "west":
        bitmap[2 * curr_task["gridsz_num_rows"] * curr_task["gridsz_num_cols"] + 1] = 1
    elif curr_task["pregrid_agent_dir"] == "south":
        bitmap[2 * curr_task["gridsz_num_rows"] * curr_task["gridsz_num_cols"] + 2] = 1
    elif curr_task["pregrid_agent_dir"] == "east":
        bitmap[2 * curr_task["gridsz_num_rows"] * curr_task["gridsz_num_cols"] + 3] = 1

    # Pre markers
    for marker in curr_task["pregrid_markers"]:
        bitmap[2 * curr_task["gridsz_num_rows"] * curr_task["gridsz_num_cols"] + 4 + marker[0] * curr_task[
            "gridsz_num_rows"] +
               marker[1]] = 1

    # Post agent loc
    bitmap[3 * curr_task["gridsz_num_rows"] * curr_task["gridsz_num_cols"] + 4 +
           curr_task["postgrid_agent_row"] * curr_task["gridsz_num_rows"] +
           curr_task["postgrid_agent_col"]] = 1

    # Agent Orientation
    if curr_task["postgrid_agent_dir"] == "north":
        bitmap[4 * curr_task["gridsz_num_rows"] * curr_task["gridsz_num_cols"] + 4 + 0] = 1
    elif curr_task["postgrid_agent_dir"] == "west":
        bitmap[4 * curr_task["gridsz_num_rows"] * curr_task["gridsz_num_cols"] + 4 + 1] = 1
    elif curr_task["postgrid_agent_dir"] == "south":
        bitmap[4 * curr_task["gridsz_num_rows"] * curr_task["gridsz_num_cols"] + 4 + 2] = 1
    elif curr_task["postgrid_agent_dir"] == "east":
        bitmap[4 * curr_task["gridsz_num_rows"] * curr_task["gridsz_num_cols"] + 4 + 3] = 1

    # Post markers
    for marker in curr_task["postgrid_markers"]:
        bitmap[4 * curr_task["gridsz_num_rows"] * curr_task["gridsz_num_cols"] + 2 * 4 +
               marker[0] * curr_task["gridsz_num_rows"] + marker[1]] = 1

    return bitmap
