from __future__ import annotations

import torch
from kuramoto.perturbations import Perturbation
from typing import List, Sequence, Tuple, Optional


def simulate_trajectory_grid(controller, theta_init, omegas, perturbations, use_filter, STEPS, DT, ROWS, COLS, N_AGENTS, K_COORD):
    theta = theta_init.clone().view(1, -1)
    history_r = []
    history_u = []
    history_theta = []
    m_hist = [] 
    r_prev = torch.abs(torch.mean(torch.exp(1j * theta), dim=1))

    curr_theta = theta
    
    # Initialize dtheta to avoid UnboundLocalError on the first iteration
    dtheta = torch.zeros_like(curr_theta) 

    # Time loop (Manual to allow BPTT - Backpropagation Through Time)
    for step in range(STEPS):
        t = step * DT

        # 1. Controller observes state and decides U
        u_control_raw, m = controller(curr_theta)

        if use_filter:
            error_sync = (1.0 - r_prev)
            gate = torch.tanh(error_sync * 10.0)
            u_control = u_control_raw * gate
        else:
            u_control = u_control_raw

        # 2. Physical Dynamics (Kuramoto + Perturbation + Control)
        # Grid reconstruction for neighbors
        th_grid = curr_theta.view(-1, ROWS, COLS)
        th_up = torch.roll(th_grid, 1, 1); th_down = torch.roll(th_grid, -1, 1)
        th_left = torch.roll(th_grid, 1, 2); th_right = torch.roll(th_grid, -1, 2)

        interaction = (torch.sin(th_up - th_grid) + torch.sin(th_down - th_grid) +
                       torch.sin(th_left - th_grid) + torch.sin(th_right - th_grid))
        interaction = interaction.view(-1, N_AGENTS)

        # Perturbation
        perturbation = torch.zeros_like(curr_theta)
        for pert in perturbations:
            if t >= pert.t0 and t <= pert.tf:
                perturbation = pert.apply(
                    t=t, theta=curr_theta, dtheta=dtheta,
                    omegas=omegas, inter=interaction, K=K_COORD
                )

        # Differential Equation
        dtheta = omegas + (K_COORD / 4.0) * interaction + perturbation + u_control

        # 3. Integration (Euler)
        curr_theta = curr_theta + dtheta * DT

        # 4. Save data for metrics
        r_now = torch.abs(torch.mean(torch.exp(1j * curr_theta), dim=1))
        r_prev = r_now
        history_r.append(r_now)
        history_u.append(u_control)
        history_theta.append(curr_theta)
        m_hist.append(m)

    return history_r, history_u, history_theta, m_hist


def simulate_trajectory_grid_disconected(controller, theta_init, omegas, perturbations, use_filter, STEPS, DT, ROWS, COLS, N_AGENTS, K_COORD):
    theta = theta_init.clone().view(1, -1)
    history_r = []
    history_u = []
    history_theta = []
    m_hist = [] 
    r_prev = torch.abs(torch.mean(torch.exp(1j * theta), dim=1))

    curr_theta = theta
    
    # Initialize dtheta to avoid UnboundLocalError on the first iteration
    dtheta = torch.zeros_like(curr_theta) 

    # Time loop (Manual to allow BPTT - Backpropagation Through Time)
    for step in range(STEPS):
        t = step * DT

        # 1. Controller observes state and decides U
        u_control_raw, m = controller(curr_theta)

        if use_filter:
            error_sync = (1.0 - r_prev)
            gate = torch.tanh(error_sync * 10.0)
            u_control = u_control_raw * gate
        else:
            u_control = u_control_raw

        # 2. Physical Dynamics (Kuramoto + Perturbation + Control)
        # Grid reconstruction for neighbors
        th_grid = curr_theta.view(-1, ROWS, COLS)
        th_up = torch.roll(th_grid, 1, 1); th_down = torch.roll(th_grid, -1, 1)
        th_left = torch.roll(th_grid, 1, 2); th_right = torch.roll(th_grid, -1, 2)

        interaction = (torch.sin(th_up - th_grid) + torch.sin(th_down - th_grid) +
                       torch.sin(th_left - th_grid) + torch.sin(th_right - th_grid))
        interaction = interaction.view(-1, N_AGENTS)

        # Perturbation especial para desconexión: setea a 0 la interacción de los nodos desconectados
        # requiere el controlo para anular la dinámica de los nodos desconectados
        perturbation = torch.zeros_like(curr_theta)
        for pert in perturbations:
            if t >= pert.t0 and t <= pert.tf:
                perturbation = pert.apply(
                    t=t, theta=curr_theta, dtheta=dtheta,
                    omegas=omegas, inter=interaction, K=K_COORD, u_control=u_control.view(-1, N_AGENTS)
                )

        # Differential Equation
        dtheta = omegas + (K_COORD / 4.0) * interaction + perturbation + u_control

        # 3. Integration (Euler)
        curr_theta = curr_theta + dtheta * DT

        # 4. Save data for metrics
        r_now = torch.abs(torch.mean(torch.exp(1j * curr_theta), dim=1))
        r_prev = r_now
        history_r.append(r_now)
        history_u.append(u_control)
        history_theta.append(curr_theta)
        m_hist.append(m)

    return history_r, history_u, history_theta, m_hist


def simulate_trajectory_topology(controller, theta_init, omegas, perturbations, use_filter, A_tensor, STEPS, DT, ROWS, COLS, N_AGENTS, K_COORD):
    theta = theta_init.clone().view(1, -1)
    history_r = []
    history_u = []
    history_theta = []
    m_hist = []
    r_prev = torch.abs(torch.mean(torch.exp(1j * theta), dim=1))

    curr_theta = theta
    
    # Initialize dtheta to avoid UnboundLocalError on the first iteration
    dtheta = torch.zeros_like(curr_theta) 

    # Time loop (Manual to allow BPTT - Backpropagation Through Time)
    for step in range(STEPS):
        t = step * DT

        # 1. Controller observes state and decides U
        u_control_raw, m = controller(curr_theta)
        

        if use_filter:
            error_sync = (1.0 - r_prev)
            gate = torch.tanh(error_sync * 10.0)
            u_control = u_control_raw * gate
        else:
            u_control = u_control_raw

        # 2. Physical Dynamics (Kuramoto + Perturbation + Control)
        # Grid reconstruction for neighbors
        
        theta_i = curr_theta.unsqueeze(2)
        theta_j = curr_theta.unsqueeze(1)

        delta_theta = theta_j - theta_i
        sin_delta = torch.sin(delta_theta)

        weighted_sin = sin_delta * A_tensor
        interaction = weighted_sin.sum(dim=2) 
        

        # Perturbation
        perturbation = torch.zeros_like(curr_theta)
        for pert in perturbations:
            if t >= pert.t0 and t <= pert.tf:
                perturbation = pert.apply(
                    t=t, theta=curr_theta, dtheta=dtheta,
                    omegas=omegas, inter=interaction, K=K_COORD
                )

        # Differential Equation
        dtheta = omegas + (K_COORD / 4.0) * interaction + perturbation + u_control

        # 3. Integration (Euler)
        curr_theta = curr_theta + dtheta * DT

        # 4. Save data for metrics
        r_now = torch.abs(torch.mean(torch.exp(1j * curr_theta), dim=1))
        r_prev = r_now
        history_r.append(r_now)
        history_u.append(u_control)
        history_theta.append(curr_theta)
        m_hist.append(m)

    return history_r, history_u, history_theta, m_hist


def simulate_trajectory_grid_hold(
    controller,
    theta_init,
    omegas,
    perturbations,
    use_filter,
    STEPS,
    DT,
    CONTROL_DT,
    ROWS,
    COLS,
    N_AGENTS,
    K_COORD
):
    theta = theta_init.clone().view(1, -1)
    history_r = []
    history_u = []
    history_theta = []
    m_hist = []

    r_prev = torch.abs(torch.mean(torch.exp(1j * theta), dim=1))
    curr_theta = theta
    dtheta = torch.zeros_like(curr_theta)

    # how many integration steps between control updates
    control_every_n_steps = max(1, int(round(CONTROL_DT / DT)))

    # initialize held control and held mask
    held_u = torch.zeros_like(curr_theta)
    held_m = torch.zeros_like(curr_theta)

    for step in range(STEPS):
        t = step * DT

        # --------------------------------------------------
        # 1. Update control only every CONTROL_DT
        # --------------------------------------------------
        if step % control_every_n_steps == 0:
            u_control_raw, m = controller(curr_theta)

            if use_filter:
                error_sync = (1.0 - r_prev)
                gate = torch.tanh(error_sync * 10.0)
                u_control = u_control_raw * gate
            else:
                u_control = u_control_raw

            held_u = u_control
            held_m = m

        # --------------------------------------------------
        # 2. Physical Dynamics
        # --------------------------------------------------
        th_grid = curr_theta.view(-1, ROWS, COLS)
        th_up = torch.roll(th_grid, 1, 1)
        th_down = torch.roll(th_grid, -1, 1)
        th_left = torch.roll(th_grid, 1, 2)
        th_right = torch.roll(th_grid, -1, 2)

        interaction = (
            torch.sin(th_up - th_grid)
            + torch.sin(th_down - th_grid)
            + torch.sin(th_left - th_grid)
            + torch.sin(th_right - th_grid)
        )
        interaction = interaction.view(-1, N_AGENTS)

        perturbation = torch.zeros_like(curr_theta)
        for pert in perturbations:
            if t >= pert.t0 and t <= pert.tf:
                perturbation = pert.apply(
                    t=t,
                    theta=curr_theta,
                    dtheta=dtheta,
                    omegas=omegas,
                    inter=interaction,
                    K=K_COORD
                )

        dtheta = omegas + (K_COORD / 4.0) * interaction + perturbation + held_u

        # --------------------------------------------------
        # 3. Integration
        # --------------------------------------------------
        curr_theta = curr_theta + dtheta * DT

        # --------------------------------------------------
        # 4. Save
        # --------------------------------------------------
        r_now = torch.abs(torch.mean(torch.exp(1j * curr_theta), dim=1))
        r_prev = r_now

        history_r.append(r_now)
        history_u.append(held_u)
        history_theta.append(curr_theta)
        m_hist.append(held_m)

    return history_r, history_u, history_theta, m_hist