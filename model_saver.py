import pandas as pd
import os


class Model_Save:
    def __init__(
        self,
        iteration: int,
        interior_loss: float,
        boundary_loss: float,
        interior_validation_loss: float,
        boundary_validation_loss: float,
        interior_point_count: int,
        boundary_point_count: int,
        time_elapsed: float,
    ):
        self.iteration = iteration
        self.interior_loss = interior_loss
        self.interior_validation_loss = interior_validation_loss
        self.boundary_loss = boundary_loss
        self.boundary_validation_loss = boundary_validation_loss
        self.interior_point_count = interior_point_count
        self.boundary_point_count = boundary_point_count
        self.time_elapsed = time_elapsed
        
    def __repr__(self):
        return f'Iteration: {self.iteration}, Interior loss: {self.interior_loss}, ' \
               f'Interior validation loss: {self.interior_validation_loss}, ' \
               f'Boundary loss: {self.boundary_loss}, ' \
               f'Boundary validation loss: {self.boundary_validation_loss}, ' \
               f'point count: {self.interior_point_count}, ' \
               f'Boundary point count: {self.boundary_point_count}, Time elapsed: {self.time_elapsed}'
    
    @staticmethod
    def save_history(history: list, L2_error: float, max_error: float, path: str):
        iters = []
        loss_int_list = []
        loss_int_validate_list = []
        boundary_loss_list = []
        boundary_loss_validate_list = []
        interior_point_counts = []
        boundary_point_counts = []
        time_elapsed = []

        for history_class in history:
            # history is a list; each element is a class History
            # initialized at each training iteration

            # append the values to the corresponding list
            iters.append(history_class.iteration)
            loss_int_list.append(history_class.interior_loss)
            loss_int_validate_list.append(history_class.interior_validation_loss)
            boundary_loss_list.append(history_class.boundary_loss)
            boundary_loss_validate_list.append(history_class.boundary_validation_loss)
            interior_point_counts.append(history_class.interior_point_count)
            boundary_point_counts.append(history_class.boundary_point_count)
            time_elapsed.append(history_class.time_elapsed)

        df = pd.DataFrame()
        df["iterations"] = iters
        df["interior_loss"] = loss_int_list
        df["interior_loss_validation"] = loss_int_validate_list
        df["boundary_loss"] = boundary_loss_list
        df["boundary_loss_validation"] = boundary_loss_validate_list
        df["interior_point_counts"] = interior_point_counts
        df["boundary_point_counts"] = boundary_point_counts
        df["time_elapsed"] = time_elapsed
        df["L2_errpr"] = L2_error
        df["max_error"] = max_error

        df.to_csv(path, index=False)


def make_path(option_name, model_name):
    os.makedirs(f"{option_name}/{model_name}/", exist_ok=True)
    os.makedirs(f"{option_name}/{model_name}/histories", exist_ok=True)

