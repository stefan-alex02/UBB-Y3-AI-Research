package ro.ubb.ai.javaserver.controller;

import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import ro.ubb.ai.javaserver.dto.user.JwtResponse;
import ro.ubb.ai.javaserver.dto.user.LoginRequest;
import ro.ubb.ai.javaserver.dto.user.RegisterRequest;
import ro.ubb.ai.javaserver.dto.user.UserDTO;
import ro.ubb.ai.javaserver.security.JwtTokenProvider;
import ro.ubb.ai.javaserver.service.UserService;

@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthenticationManager authenticationManager;
    private final UserService userService;
    private final JwtTokenProvider jwtTokenProvider;

    @PostMapping("/login")
    public ResponseEntity<?> authenticateUser(@Valid @RequestBody LoginRequest loginRequest) {
        Authentication authentication = authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(loginRequest.getUsername(), loginRequest.getPassword()));

        SecurityContextHolder.getContext().setAuthentication(authentication);
        String jwt = jwtTokenProvider.generateToken(authentication);

        UserDetails userDetails = (UserDetails) authentication.getPrincipal();

        ro.ubb.ai.javaserver.entity.User domainUser = jwtTokenProvider.getUserFromAuthentication(authentication);


        return ResponseEntity.ok(new JwtResponse(jwt,
                domainUser.getId(),
                userDetails.getUsername(),
                domainUser.getRole()
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
