package ro.ubb.ai.javaserver.service;

import ro.ubb.ai.javaserver.dto.user.RegisterRequest;
import ro.ubb.ai.javaserver.dto.user.UserDTO;
import ro.ubb.ai.javaserver.dto.user.UserUpdateRequest;
import ro.ubb.ai.javaserver.entity.User;
import ro.ubb.ai.javaserver.enums.Role;
import ro.ubb.ai.javaserver.exception.ResourceNotFoundException;
import ro.ubb.ai.javaserver.repository.UserRepository;
import ro.ubb.ai.javaserver.util.MeteorologistPasscodeValidator; // Assuming you create this
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
@Slf4j
public class UserServiceImpl implements UserService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final MeteorologistPasscodeValidator passcodeValidator; // Inject this

    @Override
    @Transactional
    public UserDTO registerUser(RegisterRequest registerRequest) {
        if (userRepository.existsByUsername(registerRequest.getUsername())) {
            throw new IllegalArgumentException("Error: Username is already taken!");
        }

        User user = new User();
        user.setUsername(registerRequest.getUsername());
        user.setPassword(passwordEncoder.encode(registerRequest.getPassword()));
        user.setName(registerRequest.getName());

        if (registerRequest.getRole() == Role.METEOROLOGIST) {
            if (!passcodeValidator.isValid(registerRequest.getMeteorologistPasscode())) {
                throw new IllegalArgumentException("Invalid meteorologist passcode.");
            }
            user.setRole(Role.METEOROLOGIST);
        } else {
            user.setRole(Role.NORMAL);
        }

        User savedUser = userRepository.save(user);
        log.info("User registered successfully: {}", savedUser.getUsername());
        return convertToDTO(savedUser);
    }

    @Override
    public UserDTO getCurrentUser() {
        String username = SecurityContextHolder.getContext().getAuthentication().getName();
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new UsernameNotFoundException("User Not Found with username: " + username));
        return convertToDTO(user);
    }

    @Override
    @Transactional
    public UserDTO updateUser(Long userId, UserUpdateRequest updateRequest) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new ResourceNotFoundException("User", "id", userId));

        // Basic authorization check (user can only update themselves, or admin can update anyone)
        String currentUsername = SecurityContextHolder.getContext().getAuthentication().getName();
        if (!user.getUsername().equals(currentUsername)) {
            // Add admin role check if needed: && !SecurityContextHolder.getContext().getAuthentication().getAuthorities().contains("ROLE_ADMIN")
            throw new SecurityException("User not authorized to update this profile.");
        }

        if (updateRequest.getName() != null && !updateRequest.getName().isBlank()) {
            user.setName(updateRequest.getName());
        }
        if (updateRequest.getNewPassword() != null && !updateRequest.getNewPassword().isBlank()) {
            user.setPassword(passwordEncoder.encode(updateRequest.getNewPassword()));
        }

        User updatedUser = userRepository.save(user);
        log.info("User profile updated for: {}", updatedUser.getUsername());
        return convertToDTO(updatedUser);
    }

    private UserDTO convertToDTO(User user) {
        UserDTO dto = new UserDTO();
        dto.setId(user.getId());
        dto.setUsername(user.getUsername());
        dto.setName(user.getName());
        dto.setRole(user.getRole());
        dto.setCreatedAt(user.getCreatedAt());
        return dto;
    }
}
