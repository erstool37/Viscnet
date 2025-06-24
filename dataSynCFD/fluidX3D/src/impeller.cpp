void main_setup() { // water stirir; required extensions: FP16S, MOVING_BOUNDARIES, SUBGRID, INTERACTIVE_GRAPHICS or GRAPHICS, VOLUME_FORCE, EQUILIBRIUM_BOUNDARIES, SURFACE
   // ################################################################## define simulation box size, viscosity, and volume force ###################################################################
   // real world parameters (SI-unit)
   const float si_box = 1.0f;
   const float fan_ratio = 0.6f;
   const float si_omega = 4.0f * 3.14f; 
   const float si_radius = si_box*fan_ratio / 2.0f; // rotor size
   const float si_u = si_radius * si_omega;
   const float si_rho = 1000.0f;
   const float si_nu = 0.000001f; // water: 0.000001
   const float si_sigma = 0.072f;
   const float si_g = 9.8f;
   
   // lbm reference values
   const uint fps = 2000u;
   const uint N = 128u;
   const uint3 lbm_grid = uint3(N, N, N); // Simulation spatial resolution
   const ulong lbm_dt = 10ull; // Simulation time resolution
   const ulong lbm_T = 480000ull;
   const float lbm_radius = (float)N * fan_ratio / 2.0f;
   const float lbm_u = lbm_radius*si_omega/fps; // lbm_u = (displacement in grids) / (time variance in # of dt steps)
   const float lbm_rho = 1.0f;   // should be set to 1.0 according to developers\

   units.set_m_kg_s(lbm_radius, lbm_u, lbm_rho, si_radius, si_u, si_rho);

   // lbm world parameters (lbm-unit)
   const float lbm_nu = units.nu(si_nu);
   const float lbm_sigma = units.sigma(si_sigma);
   const float lbm_f = units.f(si_rho, si_g);
   const float lbm_omega = units.omega(si_omega), lbm_domega = lbm_omega * lbm_dt;

   //LBM lbm(128u, 128u, 128u, 0.001f, 0.0f, 0.0f, -0.0001f, 0.001f); // grid(uint3), nu, fx, fy, fz, sigma 
   LBM lbm(lbm_grid, lbm_nu, 0.0f, 0.0f, -lbm_f, lbm_sigma);
   const float3 center = float3(lbm.center().x, lbm.center().y, 0.2f*lbm_radius);   

   // ###################################################################################### define geometry ######################################################################################
   Mesh* mesh = read_stl(get_exe_path() + "../stl/fan.stl", lbm.size(), center, 2.0f*lbm_radius);
   const uint Nx = lbm.get_Nx(), Ny = lbm.get_Ny(), Nz = lbm.get_Nz(); 

   parallel_for(lbm.get_N(), [&](ulong n) {
      uint x = 0u, y = 0u, z = 0u;
      lbm.coordinates(n, x, y, z);
      if (z < Nz / 3.0f) lbm.flags[n] = TYPE_F;
      //if (cylinder(x, y, z, center, float3(0.0f, 0.0f, 100.0f), lbm.get_Nx()/2u) == 0) {
      //   lbm.flags[n] = TYPE_S;
      //}
      if (x == 0u || x == Nx - 1u || y == 0u || y == Ny - 1u || z == 0u || z == Nz - 1u)lbm.flags[n] = TYPE_S;
      if (lbm.flags[n] != TYPE_S && lbm.flags[n] != (TYPE_S | TYPE_X)) lbm.u.y[n] = 0.001f; 
   
   });

   // ####################################################################### run simulation, export images and data ##########################################################################
   lbm.graphics.visualization_modes = VIS_PHI_RAYTRACE ;  //VIS_FLAG_LATTICE  //| VIS_Q_CRITERION;
   lbm.run(0u, lbm_T); // Initialize simulation

   while (lbm.get_t() < lbm_T) { // Main simulation loop
      
      lbm.voxelize_mesh_on_device(mesh, TYPE_S | TYPE_X, center, float3(0.0f), float3(0.0f, 0.0f, -lbm_omega)); // Rotate fan in water
      mesh->rotate(float3x3(float3(0.0f, 0.0f, 1.0f), -lbm_domega)); // Rotate mesh
      lbm.run(lbm_dt, lbm_T);

#if defined(GRAPHICS) && !defined(INTERACTIVE_GRAPHICS)
      if (lbm.graphics.next_frame(lbm_T, 30.0f)) {
         lbm.graphics.set_camera_free(float3(0.353512f * (float)Nx, -0.150326f * (float)Ny, 1.643939f * (float)Nz), -25.0f, 61.0f, 100.0f);
         lbm.graphics.write_frame();
      }
#endif // GRAPHICS && !INTERACTIVE_GRAPHICS
   }
} /**/