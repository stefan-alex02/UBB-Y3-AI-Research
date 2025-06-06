package ro.ubb.ai.javaserver.controller;

import ro.ubb.ai.javaserver.dto.user.UserDTO;
import ro.ubb.ai.javaserver.dto.user.UserUpdateRequest;
import ro.ubb.ai.javaserver.service.UserService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
public class UserController {

    private final UserService userService;

    @GetMapping("/me")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<UserDTO> getCurrentUser() {
        return ResponseEntity.ok(userService.getCurrentUser());
    }

    @PutMapping("/{id}/settings")
    @PreAuthorize("isAuthenticated() and #id == authentication.principal.id") // User can only update their own settings
    // If UserDetails principal doesn't directly have 'id', you'd fetch User by username from principal first.
    // Or, simplify and just use /me/settings and get ID from authenticated principal in service.
    public ResponseEntity<UserDTO> updateUserSettings(@PathVariable Long id, @Valid @RequestBody UserUpdateRequest updateRequest) {
        // Consider changing endpoint to /api/users/me/settings and get ID from context in service
        UserDTO updatedUser = userService.updateUser(id, updateRequest);
        return ResponseEntity.ok(updatedUser);
    }
}
