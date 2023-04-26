class SymbolicKarel:
    """ Class of Basic Karel Environment. Given symbolic representation of Karel environment executes 
    given action and returns the updated symbolic state representation """
    def __init__(self, symbolic_task):

        # Maintain current grid symbolically
        self.curr_grid = symbolic_task
        self.n_walls = len(self.curr_grid["walls"])
        self.actions = {0: "move", 1: "turn_left", 2: "turn_right", 3: "pick_marker", 4: "finish", 5: "put_marker"}
        self.n_actions = len(self.actions)
        self.crash = False
        self.finish = False
        self.no_marker = False

    def draw(self):

        curr_env = [["." for x in range(self.curr_grid["gridsz_num_rows"])] for y in range(self.curr_grid["gridsz_num_cols"])]
        for i in range(len(self.curr_grid["pregrid_markers"])):
            curr_env[self.curr_grid["pregrid_markers"][i][0]][self.curr_grid["pregrid_markers"][i][1]] = "o"
        for i in range(self.n_walls):
            curr_env[self.curr_grid["walls"][i][0]][self.curr_grid["walls"][i][1]] = "#"
        if [self.curr_grid["pregrid_agent_row"], self.curr_grid["pregrid_agent_col"]] in self.curr_grid["pregrid_markers"]:
            curr_env[self.curr_grid["pregrid_agent_row"]][self.curr_grid["pregrid_agent_col"]] = self.curr_grid["pregrid_agent_dir"][0].upper()
        else:
            curr_env[self.curr_grid["pregrid_agent_row"]][self.curr_grid["pregrid_agent_col"]] = self.curr_grid["pregrid_agent_dir"][0]

        for i in curr_env:
            print(*i)
        print('\n')

    def _front_is_clear(self):
        fut_x, fut_y = self._next_position()
        if fut_x >= self.curr_grid["gridsz_num_rows"] or fut_x < 0 or fut_y < 0 \
                or fut_y >= self.curr_grid["gridsz_num_cols"] or [fut_x, fut_y] in self.curr_grid["walls"]:
            return False
        return True

    def _next_position(self):
        curx = self.curr_grid["pregrid_agent_row"]
        cury = self.curr_grid["pregrid_agent_col"]
        #South
        if self.curr_grid["pregrid_agent_dir"].casefold() == "south":
            curx = curx + 1
        #North
        elif self.curr_grid["pregrid_agent_dir"].casefold() == "north":
            curx = curx - 1
        #East
        elif self.curr_grid["pregrid_agent_dir"].casefold() == "east":
            cury = cury + 1
        #West
        elif self.curr_grid["pregrid_agent_dir"].casefold() == "west":
            cury = cury - 1
        return curx, cury

    def action_in_grid(self, action):
        # Move
        if self.actions[action] == "move":
            # Condition for crashing
            if not self._front_is_clear():
                self.crash = True
            if self.curr_grid["pregrid_agent_dir"].casefold() == "north" and self._front_is_clear():
                self.curr_grid["pregrid_agent_row"] = self.curr_grid["pregrid_agent_row"] - 1

            if self.curr_grid["pregrid_agent_dir"].casefold() == 'east' and self._front_is_clear():
                self.curr_grid["pregrid_agent_col"] = self.curr_grid["pregrid_agent_col"] + 1

            if self.curr_grid["pregrid_agent_dir"].casefold() == "south" and self._front_is_clear():
                self.curr_grid["pregrid_agent_row"] = self.curr_grid["pregrid_agent_row"] + 1

            if self.curr_grid["pregrid_agent_dir"].casefold() == "west" and self._front_is_clear():
                self.curr_grid["pregrid_agent_col"] = self.curr_grid["pregrid_agent_col"] - 1

        elif self.actions[action] == "turn_left":

            if self.curr_grid["pregrid_agent_dir"].casefold() == "north":
                self.curr_grid["pregrid_agent_dir"] = "west"

            elif self.curr_grid["pregrid_agent_dir"].casefold() == "south":
                self.curr_grid["pregrid_agent_dir"] = "east"

            elif self.curr_grid["pregrid_agent_dir"].casefold() == "east":
                self.curr_grid["pregrid_agent_dir"] = "north"

            elif self.curr_grid["pregrid_agent_dir"].casefold() == "west":
                self.curr_grid["pregrid_agent_dir"] = "south"

        elif self.actions[action] == "turn_right":

            if self.curr_grid["pregrid_agent_dir"].casefold() == "north":
                self.curr_grid["pregrid_agent_dir"] = "east"

            elif self.curr_grid["pregrid_agent_dir"].casefold() == "south":
                self.curr_grid["pregrid_agent_dir"] = "west"

            elif self.curr_grid["pregrid_agent_dir"].casefold() == "east":
                self.curr_grid["pregrid_agent_dir"] = "south"

            elif self.curr_grid["pregrid_agent_dir"].casefold() == "west":
                self.curr_grid["pregrid_agent_dir"] = "north"

        elif self.actions[action] == "pick_marker":
            if [self.curr_grid["pregrid_agent_row"], self.curr_grid["pregrid_agent_col"]] \
                    in self.curr_grid["pregrid_markers"]:
                self.curr_grid["pregrid_markers"].remove([self.curr_grid["pregrid_agent_row"],
                                                         self.curr_grid["pregrid_agent_col"]])
            else:
                self.no_marker = True

        elif self.actions[action] == "put_marker":
            self.curr_grid["pregrid_markers"].append(
                [self.curr_grid["pregrid_agent_row"], self.curr_grid["pregrid_agent_col"]])

        elif self.actions[action] == "finish":
            self.finish = True

        return self.curr_grid, self.finish, self.crash, self.no_marker
