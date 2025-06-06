package ro.ubb.ai.javaserver.controller;

import ro.ubb.ai.javaserver.dto.user.JwtResponse;
import ro.ubb.ai.javaserver.dto.user.LoginRequest;
import ro.ubb.ai.javaserver.dto.user.RegisterRequest;
import ro.ubb.ai.javaserver.dto.user.UserDTO;
import ro.ubb.ai.javaserver.entity.User; // For UserDetails
import ro.ubb.ai.javaserver.enums.Role;
import ro.ubb.ai.javaserver.security.JwtTokenProvider; // Assuming you have this
import ro.ubb.ai.javaserver.service.UserService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails; // Spring Security UserDetails
import org.springframework.web.bind.annotation.*;

import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthenticationManager authenticationManager; // Inject this
    private final UserService userService;
    private final JwtTokenProvider jwtTokenProvider; // Assuming you create this

    @PostMapping("/login")
    public ResponseEntity<?> authenticateUser(@Valid @RequestBody LoginRequest loginRequest) {
        Authentication authentication = authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(loginRequest.getUsername(), loginRequest.getPassword()));

        SecurityContextHolder.getContext().setAuthentication(authentication);
        String jwt = jwtTokenProvider.generateToken(authentication);

        // Cast Principal to your UserDetails implementation or directly to User if User implements UserDetails
        UserDetails userDetails = (UserDetails) authentication.getPrincipal();
        // Assuming your UserDetails impl or User entity has getId() and getRole()
        // If UserDetails is a custom class, cast to it. If it's Spring's User, get authorities.

        // This part depends on how your UserDetails is structured.
        // For simplicity, let's assume a way to get the User entity or its relevant fields.
        // This might involve another service call or a custom UserDetails implementation.
        ro.ubb.ai.javaserver.entity.User domainUser = jwtTokenProvider.getUserFromAuthentication(authentication);


        return ResponseEntity.ok(new JwtResponse(jwt,
                domainUser.getId(),
                userDetails.getUsername(),
                domainUser.getRole() // Assuming Role is a field in your User entity
        ));
    }

    @PostMapping("/register")
    public ResponseEntity<?> registerUser(@Valid @RequestBody RegisterRequest registerRequest) {
        try {
            UserDTO registeredUser = userService.registerUser(registerRequest);
            return ResponseEntity.ok("User registered successfully! Welcome " + registeredUser.getUsername());
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().body(e.getMessage());
        }
    }
}
